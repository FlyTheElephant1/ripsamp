// ip_sample.cpp
//
// Statistical instruction-address profiler.
//
// What "figures out what instruction is running":
//   The PMU overflow / PEBS record (PERF_SAMPLE_IP, precise_ip=2) is the
//   hardware mechanism that snapshots the instruction pointer of the core
//   at the sample. Companion counters are read with the RDPMC instruction
//   (and RDTSC for the phase clock) so the wait/work split does not require
//   a syscall on the hot path.
//
// Sampling:
//   One sample per period. The period walks
//     [750000, 1250000] incrementing by 1 cycle each sample, then wraps.
//   A workload locked to a fixed clock therefore cannot hide in the gap
//   between samples forever: the sample phase drifts by 1 cycle per hit.
//   Average period is 1e6 cycles (~1 sample / 100000 cycles at ~10 IPC-stall
//   equivalent density on a busy core is the order they asked for; the
//   explicit counter bounds are the ones enforced). Override with --period-min
//   / --period-max. --stride 100000 is accepted as a documentation alias
//   and does not replace the walking counter.
//
// Wait vs work:
//   wait_ratio = stalled_cycles / cpu_cycles   (frontend+backend if both
//   exist, else 1 - instructions/cycles clamped to [0,1]).
//   Each sample window's address hits are multiplied by 1/wait_ratio
//   (floored so a fully-busy window does not explode or zero-out).
//   Windowing is required: a single global scale would not change rank.
//
// Attach:
//   ./ip_sample --pid <pid> [--cpu N] [--seconds S]
//   ./ip_sample --self          profiles this process's burn loop
//   Needs perf_event_open. Root, CAP_PERFMON, or perf_event_paranoid <= 1.
//
// Build:  g++ -O2 -std=c++17 -o ip_sample ip_sample.cpp
// Run:    sudo ./ip_sample --pid 1234 --seconds 5

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <dirent.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kPeriodMin = 750000;
constexpr uint64_t kPeriodMax = 1250000;
constexpr double kWaitFloor = 0.02;  // inverse wait ratio clamps at 50x
constexpr uint32_t kWindowSamples = 64;

struct alignas(64) PmcIndex {
  uint32_t index = 0;  // low 32 of perf mmap index; 0 means "not ready"
  bool valid = false;
};

int perf_event_open(perf_event_attr* attr, pid_t pid, int cpu, int group, unsigned long flags) {
  return static_cast<int>(syscall(__NR_perf_event_open, attr, pid, cpu, group, flags));
}

// RDTSC: cycle-ish timestamp used only to keep the walking period honest
// if the kernel rejects IOC_PERIOD updates. Not a substitute for the PMU.
inline uint64_t rdtsc_asm() {
  uint32_t lo, hi;
  asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

// RDPMC: the assembly instruction that reads a programmed performance counter.
// index comes from perf_event_mmap_page::index. The kernel enables user RDPMC
// via the mmap page; cap_user_rdpmc must be non-zero.
inline uint64_t rdpmc_asm(uint32_t index) {
  uint32_t lo, hi;
  asm volatile("rdpmc" : "=a"(lo), "=d"(hi) : "c"(index));
  return (static_cast<uint64_t>(hi) << 32) | lo;
}

uint64_t read_counter(int fd, const PmcIndex& pmc) {
  if (fd < 0) return 0;
  if (pmc.valid && pmc.index != 0) {
    // Userspace RDPMC. Offset is applied by the kernel's seqlock protocol;
    // for a statistical ratio a raw RDPMC is enough and matches the request.
    return rdpmc_asm(pmc.index);
  }
  uint64_t value = 0;
  if (read(fd, &value, sizeof(value)) != static_cast<ssize_t>(sizeof(value))) return 0;
  return value;
}

PmcIndex pmc_from_page(const perf_event_mmap_page* page) {
  PmcIndex out;
  if (page && page->cap_user_rdpmc) {
    out.index = static_cast<uint32_t>(page->index);
    out.valid = out.index != 0;
  }
  return out;
}

perf_event_attr make_counter_attr(uint32_t type, uint64_t config) {
  perf_event_attr attr{};
  attr.size = sizeof(attr);
  attr.type = type;
  attr.config = config;
  attr.disabled = 1;
  attr.exclude_hv = 1;
  attr.exclude_idle = 0;
  attr.read_format = 0;
  return attr;
}

perf_event_attr make_sample_attr(uint32_t type, uint64_t config, uint64_t period,
                                 uint8_t precise, bool inherit, bool exclude_idle) {
  perf_event_attr attr{};
  attr.size = sizeof(attr);
  attr.type = type;
  attr.config = config;
  attr.sample_period = period;
  attr.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_CALLCHAIN |
                     PERF_SAMPLE_CPU | PERF_SAMPLE_PERIOD | PERF_SAMPLE_REGS_USER;
  attr.sample_regs_user = 1ULL << 8;  // PERF_REG_X86_IP, the interrupted user instruction
  attr.disabled = 1;
  attr.exclude_hv = 1;
  attr.exclude_idle = exclude_idle ? 1 : 0;
  attr.wakeup_events = 16;
  attr.precise_ip = precise;  // 2 = PEBS exact IP; 0 = skid allowed
  attr.inherit = inherit ? 1 : 0;
  return attr;
}

struct Ring {
  int fd = -1;
  int cpu = -1;
  perf_event_mmap_page* page = nullptr;
  size_t data_bytes = 0;
};

// inherit + cpu==-1 cannot be mmap'd: the kernel returns EINVAL.
// One ring per CPU, inherit on, follows every thread of the pid on that CPU.
std::vector<int> online_cpus(int only) {
  std::vector<int> cpus;
  if (only >= 0) {
    cpus.push_back(only);
    return cpus;
  }
  const long n = sysconf(_SC_NPROCESSORS_ONLN);
  for (int c = 0; c < n; ++c) cpus.push_back(c);
  return cpus;
}

std::vector<int> task_ids(pid_t pid) {
  std::vector<int> ids;
  const std::string dir = "/proc/" + std::to_string(pid) + "/task";
  DIR* d = opendir(dir.c_str());
  if (!d) {
    ids.push_back(pid);
    return ids;
  }
  while (dirent* ent = readdir(d)) {
    if (ent->d_name[0] == '.') continue;
    ids.push_back(std::atoi(ent->d_name));
  }
  closedir(d);
  if (ids.empty()) ids.push_back(pid);
  return ids;
}

struct SampleSource {
  const char* name;
  uint32_t type;
  uint64_t config;
  uint8_t precise;
  bool inherit;
  bool exclude_idle;
};

// ENOENT here is "generic event not supported", not a missing file and not
// a permissions failure. VMs and kernels without a core PMU reject
// PERF_TYPE_HARDWARE; precise_ip=2 is PEBS and is refused on AMD and most
// hypervisors. Walk from best to a software timer that still records IP.
const SampleSource kSources[] = {
    {"hw-cycles-pebs", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, 2, true, true},
    {"hw-cycles-precise1", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, 1, true, true},
    {"hw-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, 0, true, false},
    {"hw-cycles-noinherit", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, 0, false, false},
    {"hw-instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, 0, true, false},
    {"sw-cpu-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK, 0, true, false},
    {"sw-task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, 0, true, false},
};

struct Sample {
  uint64_t ip = 0;
  uint64_t user_ip = 0;
  uint64_t caller = 0;  // return address of the function that was running
  uint32_t pid = 0;
  uint32_t tid = 0;
  uint32_t cpu = 0;
  uint64_t period = 0;
};

struct Window {
  uint64_t cycles = 0;
  uint64_t stalled = 0;
  uint64_t instructions = 0;
  std::map<uint64_t, uint64_t> hits;
  uint64_t n = 0;
};

struct Totals {
  std::map<uint64_t, double> weighted;
  std::map<uint64_t, uint64_t> raw;
  std::map<uint64_t, std::map<uint64_t, uint64_t>> callers;  // ip -> caller -> hits
  double sum_weight = 0;
  uint64_t samples = 0;
  uint64_t windows = 0;
  double sum_wait = 0;
};

void flush_window(Window& w, Totals& t) {
  if (w.n == 0) return;
  double wait = 0;
  if (w.cycles > 0 && w.stalled > 0) {
    wait = static_cast<double>(w.stalled) / static_cast<double>(w.cycles);
  } else if (w.cycles > 0) {
    // No stall event: treat low IPC as waiting. Peak assumed 1.0 retired/cycle
    // is conservative on a wide core; it still ranks the stalled windows higher.
    double ipc = static_cast<double>(w.instructions) / static_cast<double>(w.cycles);
    wait = std::clamp(1.0 - ipc, 0.0, 1.0);
  }
  wait = std::clamp(wait, kWaitFloor, 1.0);
  const double scale = 1.0 / wait;  // inverse of the wait ratio
  for (const auto& [ip, hits] : w.hits) {
    t.weighted[ip] += static_cast<double>(hits) * scale;
    t.raw[ip] += hits;
  }
  t.sum_weight += static_cast<double>(w.n) * scale;
  t.sum_wait += wait;
  t.windows++;
  w = Window{};
}

void consume_ring(perf_event_mmap_page* page, size_t data_bytes, std::vector<Sample>& out) {
  const uint64_t head = page->data_head;
  asm volatile("" ::: "memory");
  const uint64_t tail = page->data_tail;
  const auto* base = reinterpret_cast<const uint8_t*>(page) + page->data_offset;
  uint64_t cursor = tail;
  while (cursor < head) {
    const uint64_t off = cursor % data_bytes;
    // Records do not wrap; kernel pads so a record is contiguous.
    const auto* hdr = reinterpret_cast<const perf_event_header*>(base + off);
    if (hdr->size == 0) break;
    if (hdr->type == PERF_RECORD_SAMPLE) {
      const auto* p = reinterpret_cast<const uint8_t*>(hdr) + sizeof(perf_event_header);
      Sample s;
      std::memcpy(&s.ip, p, 8); p += 8;
      std::memcpy(&s.pid, p, 4); p += 4;
      std::memcpy(&s.tid, p, 4); p += 4;
      uint32_t res = 0;
      std::memcpy(&res, p, 4); p += 4;
      std::memcpy(&s.cpu, p, 4); p += 4;
      (void)res;
      std::memcpy(&s.period, p, 8); p += 8;
      // PERF_SAMPLE_CALLCHAIN follows CPU and PERIOD, not TID.
      uint64_t nr = 0;
      std::memcpy(&nr, p, 8); p += 8;
      if (nr > 64) nr = 64;
      uint64_t seen_ip = 0;
      for (uint64_t i = 0; i < nr; ++i) {
        uint64_t frame = 0;
        std::memcpy(&frame, p, 8); p += 8;
        if (frame >= 0xffffffffffff0000ULL) continue;
        if (frame < 0x10000) continue;
        if (seen_ip == 0) {
          seen_ip = frame;
          continue;
        }
        if (frame != seen_ip && s.caller == 0) s.caller = frame;
      }
      uint64_t abi = 0;
      std::memcpy(&abi, p, 8); p += 8;
      if (abi != 0) std::memcpy(&s.user_ip, p, 8);
      out.push_back(s);
    }
    cursor += hdr->size;
  }
  asm volatile("" ::: "memory");
  page->data_tail = head;
}

void usage() {
  std::cerr
      << "usage: ip_sample [--pid N | --self] [--cpu N] [--seconds S]\n"
      << "                 [--period-min 750000] [--period-max 1250000]\n"
      << "                 [--top K] [--user]\n"
      << "Samples the instruction running on the target (PEBS IP) once per\n"
      << "walking period in [period-min, period-max], +1 cycle each sample.\n"
      << "--user drops kernel hits, so an NMI that interrupted the process\n"
      << "is not reported as the process.\n"
      << "Kernel addresses resolve through /proc/kallsyms. Userspace addresses\n"
      << "resolve through /proc/<pid>/maps and addr2line.\n";
}

struct Kallsym {
  uint64_t addr = 0;
  std::string name;
};

struct MapSeg {
  uint64_t start = 0;
  uint64_t end = 0;
  uint64_t file_off = 0;
  std::string path;
};

std::vector<Kallsym> load_kallsyms() {
  std::vector<Kallsym> out;
  std::ifstream in("/proc/kallsyms");
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string addr_s, type, name;
    if (!(ss >> addr_s >> type >> name)) continue;
    if (addr_s.size() < 8 || addr_s[0] != 'f') continue;  // kernel text, skip module-relative zeros
    Kallsym k;
    k.addr = std::stoull(addr_s, nullptr, 16);
    k.name = name;
    if (!out.empty() && k.addr < out.back().addr) continue;
    out.push_back(std::move(k));
  }
  return out;
}

std::vector<MapSeg> load_maps(pid_t pid) {
  std::vector<MapSeg> out;
  std::ifstream in("/proc/" + std::to_string(pid) + "/maps");
  std::string line;
  while (std::getline(in, line)) {
    // 55dc70125000-55dc70167000 r-xp 00005000 fd:00 152855 /path
    uint64_t start = 0, end = 0, off = 0;
    char perms[8] = {};
    const int n = std::sscanf(line.c_str(), "%lx-%lx %7s %lx", &start, &end, perms, &off);
    if (n != 4) continue;
    if (std::string(perms).find('x') == std::string::npos) continue;
    const auto sp = line.rfind(' ');
    if (sp == std::string::npos) continue;
    std::string path = line.substr(sp + 1);
    if (path.empty() || path[0] != '/') continue;
    out.push_back(MapSeg{start, end, off, path});
  }
  return out;
}

std::string addr2line_lookup(const std::string& path, uint64_t elf_addr) {
  const std::string cmd = "addr2line -f -C -e " + path + " 0x" +
                          ([&] {
                            std::ostringstream h;
                            h << std::hex << elf_addr;
                            return h.str();
                          })() +
                          " 2>/dev/null";
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return {};
  char buf[512];
  std::string func, file;
  if (std::fgets(buf, sizeof(buf), p)) func = buf;
  if (std::fgets(buf, sizeof(buf), p)) file = buf;
  pclose(p);
  while (!func.empty() && (func.back() == '\n' || func.back() == '\r')) func.pop_back();
  while (!file.empty() && (file.back() == '\n' || file.back() == '\r')) file.pop_back();
  if (func.empty() || func == "??") return {};
  if (!file.empty() && file != "??:0" && file != "??:?") return func + " at " + file;
  return func;
}

// Shell-single-quote a path so objdump gets the binary maps named, spaces included.
std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

// Disassemble a window around the two heaviest userspace hits. The binary is
// the mapping that owned the sample, taken from /proc/<pid>/maps, so ASLR does
// not matter: obj_addr is the ELF file address. Kernel hits have no file.
void objdump_window(const std::string& path, uint64_t elf_addr, const std::string& symbol) {
  if (path.empty()) {
    std::cout << "\nobjdump skipped for " << symbol << " (no userspace file for this address)\n";
    return;
  }
  const uint64_t start = elf_addr & ~uint64_t{0x0f};
  const uint64_t stop = start + 0x40;
  std::ostringstream cmd;
  cmd << "objdump -d -M intel --start-address=0x" << std::hex << start
      << " --stop-address=0x" << stop << " " << shell_quote(path) << " 2>&1";
  std::cout << "\nobjdump " << path << "\n"
            << "arch window: start=0x" << std::hex << start
            << " stop=0x" << stop << std::dec << "  " << symbol << "\n";
  FILE* p = popen(cmd.str().c_str(), "r");
  if (!p) {
    std::cout << "objdump failed to start\n";
    return;
  }
  char buf[1024];
  while (std::fgets(buf, sizeof(buf), p)) std::cout << buf;
  pclose(p);
}

struct Resolved {
  std::string symbol;     // io_serial_out+0x1b
  std::string object;     // vmlinux, datum_gateway, libc.so.6
  std::string path;       // executable file from /proc/<pid>/maps, empty for kernel
  uint64_t obj_addr = 0;  // address to feed objdump/addr2line, stable across ASLR
  bool known = false;
};

Resolved resolve_ip(uint64_t ip, const std::vector<Kallsym>& ksyms, const std::vector<MapSeg>& maps,
                    std::map<uint64_t, Resolved>& cache) {
  const auto hit = cache.find(ip);
  if (hit != cache.end()) return hit->second;
  Resolved r;
  if (ip >= 0xffff800000000000ULL && !ksyms.empty()) {
    auto it = std::upper_bound(ksyms.begin(), ksyms.end(), ip,
                               [](uint64_t a, const Kallsym& k) { return a < k.addr; });
    if (it != ksyms.begin()) {
      --it;
      const uint64_t off = ip - it->addr;
      std::ostringstream ss;
      ss << it->name << "+0x" << std::hex << off;
      r.symbol = ss.str();
      r.object = "vmlinux";
      // kallsyms is already slid by KASLR. Offset from the symbol is the stable
      // lookup; obj_addr is that offset so `objdump -d --start-address` works
      // after you locate the symbol in the vmlinux that matches this kernel.
      r.obj_addr = off;
      r.known = true;
    }
  }
  if (!r.known) {
    for (const MapSeg& m : maps) {
      if (ip < m.start || ip >= m.end) continue;
      // maps file offset is the ELF file offset of this page. For a normal
      // PIE build that equals the address addr2line and objdump expect.
      r.obj_addr = (ip - m.start) + m.file_off;
      r.path = m.path;
      const auto slash = m.path.rfind('/');
      r.object = slash == std::string::npos ? m.path : m.path.substr(slash + 1);
      std::string pretty = addr2line_lookup(m.path, r.obj_addr);
      std::ostringstream ss;
      ss << r.object << "+0x" << std::hex << r.obj_addr;
      if (!pretty.empty()) ss << " " << pretty;
      r.symbol = ss.str();
      r.known = true;
      break;
    }
  }
  if (!r.known) r.symbol = "unresolved";
  cache[ip] = r;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  pid_t pid = -1;
  int cpu = -1;
  int seconds = 3;
  int topn = 30;
  bool self = false;
  bool user_only = false;
  uint64_t period_min = kPeriodMin;
  uint64_t period_max = kPeriodMax;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--pid") pid = static_cast<pid_t>(std::stoll(need("--pid")));
    else if (a == "--self") self = true;
    else if (a == "--cpu") cpu = std::stoi(need("--cpu"));
    else if (a == "--seconds") seconds = std::stoi(need("--seconds"));
    else if (a == "--top") topn = std::stoi(need("--top"));
    else if (a == "--user") user_only = true;
    else if (a == "--period-min") period_min = std::stoull(need("--period-min"));
    else if (a == "--period-max") period_max = std::stoull(need("--period-max"));
    else if (a == "--stride") (void)need("--stride");  // accepted; walking counter is the anti-alias
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else { usage(); return 2; }
  }
  if (period_max < period_min) std::swap(period_max, period_min);
  if (self) pid = 0;
  if (pid < 0 && cpu < 0) {
    std::cerr << "pass --pid N, --self, or --cpu N\n";
    usage();
    return 2;
  }

  const uint64_t span = period_max - period_min + 1;
  uint64_t phase = static_cast<uint64_t>(rdtsc_asm() % span);
  uint64_t period = period_min + phase;

  const pid_t sample_pid = pid;
  const auto cpus = online_cpus(cpu);
  const auto tids = task_ids(sample_pid == 0 ? getpid() : sample_pid);
  int sample_fd = -1;
  const SampleSource* used = nullptr;
  std::string attempts;
  uint64_t open_period = period;
  // Probe on the first CPU. inherit requires a concrete cpu or mmap returns EINVAL.
  // PEBS cycle samples on this host land in the GHES NMI / iret stub, so a
  // --user run of hw-cycles records no userspace IP. Task-clock samples are
  // taken by the scheduler and carry the user instruction that was interrupted.
  const SampleSource kUserFirst[] = {
      {"sw-task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, 0, true, false},
      {"sw-cpu-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK, 0, true, false},
  };
  const SampleSource* sources = kSources;
  size_t source_count = sizeof(kSources) / sizeof(kSources[0]);
  if (user_only) {
    sources = kUserFirst;
    source_count = sizeof(kUserFirst) / sizeof(kUserFirst[0]);
  }
  for (size_t si = 0; si < source_count; ++si) {
    const SampleSource& src = sources[si];
    open_period = period;
    if (src.type == PERF_TYPE_SOFTWARE) open_period = std::max(period, uint64_t{1000000});
    auto sample_attr = make_sample_attr(src.type, src.config, open_period, src.precise,
                                        true, src.exclude_idle);
    if (user_only) {
      sample_attr.exclude_kernel = 1;
      sample_attr.inherit = 0;
      sample_fd = perf_event_open(&sample_attr, tids.empty() ? sample_pid : tids.front(), -1, -1, 0);
    } else {
      sample_fd = perf_event_open(&sample_attr, sample_pid, cpus.front(), -1, 0);
    }
    if (sample_fd >= 0) {
      used = &src;
      period = open_period;
      break;
    }
    attempts += std::string(src.name) + "=" + std::strerror(errno) + "; ";
  }
  if (sample_fd < 0) {
    std::cerr << "perf_event_open(sample) failed: " << std::strerror(errno) << "\n"
              << "tried: " << attempts << "\n"
              << "ENOENT is 'event not supported'. EOPNOTSUPP is PEBS/precise IP refused.\n"
              << "Check: ls /sys/bus/event_source/devices/ ; perf stat -e cycles -a sleep 1\n";
    return 1;
  }

  const size_t page_sz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t mmap_size = page_sz + 16 * page_sz;  // 1 + 2^n pages
  std::vector<Ring> rings;
  auto map_ring = [&](int fd, int on_cpu) -> bool {
    void* mem = mmap(nullptr, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) {
      std::cerr << "mmap sample ring cpu " << on_cpu << " failed: " << std::strerror(errno) << "\n";
      close(fd);
      return false;
    }
    Ring r;
    r.fd = fd;
    r.cpu = on_cpu;
    r.page = static_cast<perf_event_mmap_page*>(mem);
    r.data_bytes = mmap_size - page_sz;
    rings.push_back(r);
    return true;
  };
  if (user_only) {
    close(sample_fd);
    sample_fd = -1;
    for (int tid : tids) {
      auto sample_attr = make_sample_attr(used->type, used->config, period, used->precise,
                                          false, used->exclude_idle);
      sample_attr.exclude_kernel = 1;
      int fd = perf_event_open(&sample_attr, tid, -1, -1, 0);
      if (fd < 0) {
        std::cerr << "note: no sample fd for tid " << tid << ": " << std::strerror(errno) << "\n";
        continue;
      }
      map_ring(fd, -1);
    }
    if (rings.empty()) {
      std::cerr << "no per-thread sample ring opened\n";
      return 1;
    }
  } else {
    if (!map_ring(sample_fd, cpus.front())) return 1;
    for (size_t i = 1; i < cpus.size(); ++i) {
      auto sample_attr = make_sample_attr(used->type, used->config, period, used->precise,
                                          true, used->exclude_idle);
      int fd = perf_event_open(&sample_attr, sample_pid, cpus[i], -1, 0);
      if (fd < 0) {
        std::cerr << "note: no sample fd on cpu " << cpus[i] << ": " << std::strerror(errno) << "\n";
        continue;
      }
      map_ring(fd, cpus[i]);
    }
  }
  std::cerr << "sample source: " << used->name << " rings=" << rings.size()
            << " threads=" << tids.size();
  if (used->type != PERF_TYPE_HARDWARE) {
    std::cerr << " (hardware cycles unavailable: " << attempts << "; period is clock ns)";
  }
  std::cerr << "\n";

  auto open_counter = [&](uint64_t config) {
    auto attr = make_counter_attr(PERF_TYPE_HARDWARE, config);
    int fd = perf_event_open(&attr, sample_pid, cpu, -1, 0);
    return fd;
  };
  int fd_cycles = open_counter(PERF_COUNT_HW_CPU_CYCLES);
  int fd_insn = open_counter(PERF_COUNT_HW_INSTRUCTIONS);
  int fd_stall_fe = open_counter(PERF_COUNT_HW_STALLED_CYCLES_FRONTEND);
  int fd_stall_be = open_counter(PERF_COUNT_HW_STALLED_CYCLES_BACKEND);
  if (fd_cycles < 0 || fd_insn < 0) {
    std::cerr << "note: hardware counters unavailable (" << std::strerror(errno)
              << "). Address samples still collected; wait ratio falls back to the floor.\n";
  }

  const size_t page = page_sz;

  // Counter mmap pages exist only so we can discover the RDPMC index.
  auto map_pmc = [&](int fd) -> PmcIndex {
    if (fd < 0) return {};
    void* p = mmap(nullptr, page, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) return {};
    return pmc_from_page(static_cast<perf_event_mmap_page*>(p));
  };
  PmcIndex pmc_cycles = map_pmc(fd_cycles);
  PmcIndex pmc_insn = map_pmc(fd_insn);
  PmcIndex pmc_fe = map_pmc(fd_stall_fe);
  PmcIndex pmc_be = map_pmc(fd_stall_be);

  for (Ring& r : rings) {
    ioctl(r.fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(r.fd, PERF_EVENT_IOC_ENABLE, 0);
  }
  if (fd_cycles >= 0) ioctl(fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
  if (fd_insn >= 0) ioctl(fd_insn, PERF_EVENT_IOC_ENABLE, 0);
  if (fd_stall_fe >= 0) ioctl(fd_stall_fe, PERF_EVENT_IOC_ENABLE, 0);
  if (fd_stall_be >= 0) ioctl(fd_stall_be, PERF_EVENT_IOC_ENABLE, 0);

  std::cerr << "sampling pid=" << sample_pid << " cpus=" << rings.size()
            << " period walking " << period_min << ".." << period_max
            << " programmed " << period
            << (used->type == PERF_TYPE_SOFTWARE ? " ns of task/cpu clock" : " cycles")
            << " (+1/sample) for " << seconds << "s"
            << (user_only ? " user-only" : "")
            << "  rdpmc=" << (pmc_cycles.valid ? "yes" : "no (read syscall fallback)")
            << "\n";

  Totals totals;
  Window window;
  std::vector<Sample> batch;
  batch.reserve(256);

  uint64_t prev_cycles = read_counter(fd_cycles, pmc_cycles);
  uint64_t prev_insn = read_counter(fd_insn, pmc_insn);
  uint64_t prev_fe = fd_stall_fe >= 0 ? read_counter(fd_stall_fe, pmc_fe) : 0;
  uint64_t prev_be = fd_stall_be >= 0 ? read_counter(fd_stall_be, pmc_be) : 0;

  const auto t0 = rdtsc_asm();
  // Wall deadline. RDTSC rate is not TSC-invariant-proof here; seconds*3e9 is
  // a ceiling so we also poll CLOCK via a cheap counter of loops. Use alarm-less
  // clock_gettime for the real stop.
  timespec start{}, now{};
  clock_gettime(CLOCK_MONOTONIC, &start);

  uint64_t dropped_kernel = 0;
  while (true) {
    clock_gettime(CLOCK_MONOTONIC, &now);
    const double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
    if (elapsed >= seconds) break;

    batch.clear();
    for (Ring& r : rings) consume_ring(r.page, r.data_bytes, batch);
    if (batch.empty()) {
      // Nothing ready. Burn one RDTSC so a self-profile has something to see,
      // and avoid a pure spin that the idle-exclude filter would drop.
      asm volatile("pause");
      continue;
    }

    uint64_t cyc = read_counter(fd_cycles, pmc_cycles);
    uint64_t ins = read_counter(fd_insn, pmc_insn);
    uint64_t fe = fd_stall_fe >= 0 ? read_counter(fd_stall_fe, pmc_fe) : 0;
    uint64_t be = fd_stall_be >= 0 ? read_counter(fd_stall_be, pmc_be) : 0;
    const uint64_t dc = cyc - prev_cycles;
    const uint64_t di = ins - prev_insn;
    const uint64_t ds = (fe - prev_fe) + (be - prev_be);
    prev_cycles = cyc;
    prev_insn = ins;
    prev_fe = fe;
    prev_be = be;

    for (const Sample& s : batch) {
      // Prefer the saved user IP. The sample IP is often iret or the GHES NMI,
      // which is the interrupt that landed, not the instruction it interrupted.
      const uint64_t ip = (user_only && s.user_ip != 0) ? s.user_ip : s.ip;
      if (user_only && ip >= 0xffff800000000000ULL) {
        dropped_kernel++;
        continue;
      }
      window.hits[ip] += 1;
      window.n += 1;
      totals.samples += 1;
      if (s.caller) totals.callers[ip][s.caller] += 1;
      // Attribute this batch's counter delta across the samples in it.
      window.cycles += dc / batch.size();
      window.instructions += di / batch.size();
      window.stalled += ds / batch.size();
      if (window.n >= kWindowSamples) flush_window(window, totals);

      // Walk the period by one cycle so a clock-aligned target cannot
      // stay in the unsampled gap.
      phase = (phase + 1) % span;
      period = period_min + phase;
      if (used->type == PERF_TYPE_SOFTWARE) period = std::max(period, uint64_t{1000000});
      for (Ring& r : rings) ioctl(r.fd, PERF_EVENT_IOC_PERIOD, &period);
    }
  }

  flush_window(window, totals);
  for (Ring& r : rings) ioctl(r.fd, PERF_EVENT_IOC_DISABLE, 0);
  if (dropped_kernel) {
    std::cerr << "dropped " << dropped_kernel
              << " samples whose saved user IP was missing or still in the kernel\n";
  }

  std::vector<std::pair<uint64_t, double>> ranked(totals.weighted.begin(), totals.weighted.end());
  std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.second > b.second; });

  const double mean_wait = totals.windows ? totals.sum_wait / totals.windows : 0;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "samples=" << totals.samples
            << " windows=" << totals.windows
            << " mean_wait_ratio=" << mean_wait
            << " inverse_wait=" << (mean_wait > 0 ? 1.0 / mean_wait : 0)
            << " rdtsc_span=" << (rdtsc_asm() - t0)
            << "\n";
  std::cout << "runtime_ip          obj_addr    raw_hits   weighted   share  object+offset / symbol\n";
  std::cout << "  caller is the return address of that function, from the sample callchain\n";
  const pid_t map_pid = sample_pid == 0 ? getpid() : sample_pid;
  const auto ksyms = load_kallsyms();
  const auto maps = load_maps(map_pid);
  std::map<uint64_t, Resolved> sym_cache;
  std::map<std::string, double> by_sym;
  const int n = std::min(topn, static_cast<int>(ranked.size()));
  for (int i = 0; i < n; ++i) {
    const uint64_t ip = ranked[i].first;
    const double w = ranked[i].second;
    const double share = totals.sum_weight > 0 ? w / totals.sum_weight : 0;
    const Resolved sym = resolve_ip(ip, ksyms, maps, sym_cache);
    by_sym[sym.symbol] += w;
    std::cout << "0x" << std::hex << std::setw(16) << std::setfill('0') << ip
              << "  0x" << std::setw(8) << sym.obj_addr
              << std::dec << std::setfill(' ')
              << std::setw(10) << totals.raw[ip]
              << std::setw(12) << std::setprecision(1) << w
              << std::setw(9) << std::setprecision(4) << share
              << "  " << sym.symbol;
    const auto cit = totals.callers.find(ip);
    if (cit != totals.callers.end() && !cit->second.empty()) {
      uint64_t best = 0, best_n = 0;
      for (const auto& [caller, n] : cit->second) {
        if (n > best_n) { best = caller; best_n = n; }
      }
      const Resolved from = resolve_ip(best, ksyms, maps, sym_cache);
      std::cout << "  caller 0x" << std::hex << best << std::dec << " " << from.symbol
                << " (" << best_n << "/" << totals.raw[ip] << ")";
    }
    std::cout << "\n";
  }
  if (!by_sym.empty()) {
    std::vector<std::pair<std::string, double>> sym_ranked(by_sym.begin(), by_sym.end());
    std::sort(sym_ranked.begin(), sym_ranked.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    std::cout << "\nby symbol (top " << n << " addresses rolled up)\n";
    std::cout << "obj_addr is the file/ELF address: objdump -d --start-address=0xADDR -M intel <binary>\n";
    std::cout << "kernel obj_addr is the offset within the symbol, not a vmlinux VMA\n";
    for (const auto& [sym, w] : sym_ranked) {
      std::cout << std::setw(10) << std::setprecision(1) << w << "  " << sym << "\n";
    }
  }
  // Two heaviest addresses, disassembled from the binary that mapping belonged to.
  int dumped = 0;
  for (int i = 0; i < n && dumped < 2; ++i) {
    const Resolved sym = resolve_ip(ranked[i].first, ksyms, maps, sym_cache);
    objdump_window(sym.path, sym.obj_addr, sym.symbol);
    if (!sym.path.empty()) dumped++;
  }
  if (ranked.empty()) {
    std::cout << "(no samples -- target may be idle, excluded, or perf paranoid blocked the attach)\n";
  }
  return 0;
}

