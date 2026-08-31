#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

#ifdef __linux__

#include <fcntl.h>
#include <linux/io_uring.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace py = pybind11;

[[noreturn]] void throw_errno(const std::string& operation) {
  const int err = errno;
  throw std::runtime_error(operation + ": " + std::strerror(err) +
                           " (errno " + std::to_string(err) + ")");
}

int open_userfaultfd() {
  const int fd = static_cast<int>(
      syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK));
  if (fd < 0) throw_errno("userfaultfd");
  struct uffdio_api api {};
  api.api = UFFD_API;
  api.features = 0;
  if (ioctl(fd, UFFDIO_API, &api) < 0) {
    const int saved = errno;
    close(fd);
    errno = saved;
    throw_errno("UFFDIO_API");
  }
  return fd;
}

void probe() {
  const int fd = open_userfaultfd();
  close(fd);
}

size_t system_page_size() {
  const long value = sysconf(_SC_PAGESIZE);
  if (value <= 0) throw std::runtime_error("could not determine system page size");
  return static_cast<size_t>(value);
}

class IoUring {
 public:
  explicit IoUring(unsigned entries = 8) {
    std::memset(&params_, 0, sizeof(params_));
    fd_ = static_cast<int>(syscall(SYS_io_uring_setup, entries, &params_));
    if (fd_ < 0) throw_errno("io_uring_setup");

    sq_ring_size_ = params_.sq_off.array +
                    params_.sq_entries * sizeof(unsigned);
    cq_ring_size_ = params_.cq_off.cqes +
                    params_.cq_entries * sizeof(struct io_uring_cqe);
    if (params_.features & IORING_FEAT_SINGLE_MMAP) {
      sq_ring_size_ = cq_ring_size_ = std::max(sq_ring_size_, cq_ring_size_);
    }
    sq_ring_ = mmap(nullptr, sq_ring_size_, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQ_RING);
    if (sq_ring_ == MAP_FAILED) {
      const int saved = errno;
      sq_ring_ = nullptr;
      cleanup();
      errno = saved;
      throw_errno("mmap io_uring submission ring");
    }
    if (params_.features & IORING_FEAT_SINGLE_MMAP) {
      cq_ring_ = sq_ring_;
    } else {
      cq_ring_ = mmap(nullptr, cq_ring_size_, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_CQ_RING);
      if (cq_ring_ == MAP_FAILED) {
        const int saved = errno;
        cq_ring_ = nullptr;
        cleanup();
        errno = saved;
        throw_errno("mmap io_uring completion ring");
      }
    }
    sqes_size_ = params_.sq_entries * sizeof(struct io_uring_sqe);
    sqes_ = static_cast<struct io_uring_sqe*>(
        mmap(nullptr, sqes_size_, PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQES));
    if (sqes_ == MAP_FAILED) {
      const int saved = errno;
      sqes_ = nullptr;
      cleanup();
      errno = saved;
      throw_errno("mmap io_uring SQEs");
    }

    sq_head_ = at<unsigned>(sq_ring_, params_.sq_off.head);
    sq_tail_ = at<unsigned>(sq_ring_, params_.sq_off.tail);
    sq_mask_ = at<unsigned>(sq_ring_, params_.sq_off.ring_mask);
    sq_entries_ = at<unsigned>(sq_ring_, params_.sq_off.ring_entries);
    sq_array_ = at<unsigned>(sq_ring_, params_.sq_off.array);
    cq_head_ = at<unsigned>(cq_ring_, params_.cq_off.head);
    cq_tail_ = at<unsigned>(cq_ring_, params_.cq_off.tail);
    cq_mask_ = at<unsigned>(cq_ring_, params_.cq_off.ring_mask);
    cqes_ = at<struct io_uring_cqe>(cq_ring_, params_.cq_off.cqes);
  }

  ~IoUring() { cleanup(); }
  IoUring(const IoUring&) = delete;
  IoUring& operator=(const IoUring&) = delete;

  void read_exact(int source_fd, void* dst, size_t length, uint64_t offset) {
    if (length > std::numeric_limits<unsigned>::max()) {
      throw std::runtime_error("UFFD row is too large for one io_uring read");
    }
    const unsigned head = __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE);
    const unsigned tail = __atomic_load_n(sq_tail_, __ATOMIC_RELAXED);
    if (tail - head >= *sq_entries_) {
      throw std::runtime_error("io_uring submission queue is full");
    }
    const unsigned index = tail & *sq_mask_;
    struct io_uring_sqe* sqe = &sqes_[index];
    std::memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ;
    sqe->fd = source_fd;
    sqe->off = offset;
    sqe->addr = reinterpret_cast<uint64_t>(dst);
    sqe->len = static_cast<unsigned>(length);
    sqe->user_data = 1;
    sq_array_[index] = index;
    __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);

    int entered;
    do {
      entered = static_cast<int>(
          syscall(SYS_io_uring_enter, fd_, 1, 1, IORING_ENTER_GETEVENTS,
                  nullptr, 0));
    } while (entered < 0 && errno == EINTR);
    if (entered < 0) throw_errno("io_uring_enter");

    unsigned cq_head = __atomic_load_n(cq_head_, __ATOMIC_RELAXED);
    while (cq_head == __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE)) {
      int waited;
      do {
        waited = static_cast<int>(
            syscall(SYS_io_uring_enter, fd_, 0, 1, IORING_ENTER_GETEVENTS,
                    nullptr, 0));
      } while (waited < 0 && errno == EINTR);
      if (waited < 0) throw_errno("io_uring completion wait");
    }
    const struct io_uring_cqe cqe = cqes_[cq_head & *cq_mask_];
    __atomic_store_n(cq_head_, cq_head + 1, __ATOMIC_RELEASE);
    if (cqe.res < 0) {
      errno = -cqe.res;
      throw_errno("io_uring O_DIRECT row read");
    }
    if (static_cast<size_t>(cqe.res) != length) {
      throw std::runtime_error(
          "short io_uring O_DIRECT row read: got " +
          std::to_string(cqe.res) + ", expected " + std::to_string(length));
    }
  }

 private:
  template <typename T>
  static T* at(void* base, uint32_t offset) {
    return reinterpret_cast<T*>(static_cast<char*>(base) + offset);
  }

  void cleanup() {
    if (sqes_) munmap(sqes_, sqes_size_);
    sqes_ = nullptr;
    if (cq_ring_ && cq_ring_ != sq_ring_) munmap(cq_ring_, cq_ring_size_);
    cq_ring_ = nullptr;
    if (sq_ring_) munmap(sq_ring_, sq_ring_size_);
    sq_ring_ = nullptr;
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
  }

  int fd_ = -1;
  struct io_uring_params params_ {};
  void* sq_ring_ = nullptr;
  void* cq_ring_ = nullptr;
  struct io_uring_sqe* sqes_ = nullptr;
  size_t sq_ring_size_ = 0;
  size_t cq_ring_size_ = 0;
  size_t sqes_size_ = 0;
  unsigned* sq_head_ = nullptr;
  unsigned* sq_tail_ = nullptr;
  unsigned* sq_mask_ = nullptr;
  unsigned* sq_entries_ = nullptr;
  unsigned* sq_array_ = nullptr;
  unsigned* cq_head_ = nullptr;
  unsigned* cq_tail_ = nullptr;
  unsigned* cq_mask_ = nullptr;
  struct io_uring_cqe* cqes_ = nullptr;
};

struct Region {
  uintptr_t start = 0;
  size_t length = 0;
  size_t nbytes = 0;
  size_t row_bytes = 0;
  int num_rows = 0;
  int source_fd = -1;
  uint64_t file_offset = 0;
  std::vector<uint8_t> resident;
  std::vector<uint64_t> last_use;

  ~Region() {
    if (source_fd >= 0) close(source_fd);
  }
};

struct Request {
  std::vector<int> regions;
  std::vector<int> rows;
  size_t pages = 0;
  bool done = false;
  std::string error;
  std::mutex mutex;
  std::condition_variable cv;
};

class UffdPager {
 public:
  explicit UffdPager(uint64_t budget_bytes)
      : budget_bytes_(budget_bytes), page_size_(system_page_size()) {
    if (budget_bytes_ == 0) throw std::invalid_argument("pager budget must be positive");
    uffd_ = open_userfaultfd();
    try {
      ring_ = std::make_unique<IoUring>();
      wake_fd_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
      if (wake_fd_ < 0) throw_errno("eventfd");
      thread_ = std::thread(&UffdPager::handler_main, this);
    } catch (...) {
      if (wake_fd_ >= 0) close(wake_fd_);
      if (uffd_ >= 0) close(uffd_);
      throw;
    }
  }

  ~UffdPager() {
    stop_.store(true, std::memory_order_release);
    wake_handler();
    if (thread_.joinable()) thread_.join();
    {
      std::lock_guard<std::mutex> guard(regions_mutex_);
      for (const auto& region : regions_) {
        struct uffdio_range range {};
        range.start = region->start;
        range.len = region->length;
        ioctl(uffd_, UFFDIO_UNREGISTER, &range);
      }
      regions_.clear();
    }
    if (bounce_) free(bounce_);
    if (wake_fd_ >= 0) close(wake_fd_);
    if (uffd_ >= 0) close(uffd_);
  }

  UffdPager(const UffdPager&) = delete;
  UffdPager& operator=(const UffdPager&) = delete;

  int add_region(uintptr_t address, size_t length, size_t nbytes,
                 const std::string& path, uint64_t file_offset,
                 size_t row_bytes, int num_rows) {
    if (address % page_size_ || length % page_size_ || file_offset % page_size_ ||
        row_bytes % page_size_) {
      throw std::invalid_argument(
          "UFFD bank address, mapping length, FTW offset, and expert row size "
          "must all be page-aligned");
    }
    if (num_rows <= 0 || row_bytes == 0 ||
        row_bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(num_rows) ||
        row_bytes * static_cast<size_t>(num_rows) != nbytes || nbytes != length) {
      throw std::invalid_argument(
          "UFFD bank mapping must consist exactly of equal, page-aligned expert rows");
    }
    if (row_bytes > budget_bytes_) {
      throw std::invalid_argument(
          "UFFD pager budget is smaller than one expert-bank row");
    }
    const int source_fd = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (source_fd < 0) throw_errno("open O_DIRECT FTW bank source " + path);

    struct uffdio_register registration {};
    registration.range.start = address;
    registration.range.len = length;
    registration.mode = UFFDIO_REGISTER_MODE_MISSING;
    if (ioctl(uffd_, UFFDIO_REGISTER, &registration) < 0) {
      const int saved = errno;
      close(source_fd);
      errno = saved;
      throw_errno("UFFDIO_REGISTER");
    }
    if (!(registration.ioctls & (1ULL << _UFFDIO_COPY))) {
      struct uffdio_range range {address, length};
      ioctl(uffd_, UFFDIO_UNREGISTER, &range);
      close(source_fd);
      throw std::runtime_error("registered UFFD region does not support UFFDIO_COPY");
    }

    auto region = std::make_unique<Region>();
    region->start = address;
    region->length = length;
    region->nbytes = nbytes;
    region->row_bytes = row_bytes;
    region->num_rows = num_rows;
    region->source_fd = source_fd;
    region->file_offset = file_offset;
    region->resident.assign(num_rows, 0);
    region->last_use.assign(num_rows, 0);
    std::lock_guard<std::mutex> guard(regions_mutex_);
    const int id = static_cast<int>(regions_.size());
    try {
      regions_.push_back(std::move(region));
    } catch (...) {
      struct uffdio_range range {address, length};
      ioctl(uffd_, UFFDIO_UNREGISTER, &range);
      throw;
    }
    return id;
  }

  size_t prefetch(const std::vector<int>& regions, const std::vector<int>& rows) {
    raise_if_error();
    auto request = std::make_shared<Request>();
    request->regions = regions;
    request->rows = rows;
    {
      std::lock_guard<std::mutex> guard(request_mutex_);
      requests_.push_back(request);
    }
    wake_handler();
    std::unique_lock<std::mutex> lock(request->mutex);
    request->cv.wait(lock, [&] { return request->done; });
    if (!request->error.empty()) throw std::runtime_error(request->error);
    return request->pages;
  }

  bool is_resident(int region_id, int row) {
    std::lock_guard<std::mutex> regions_guard(regions_mutex_);
    Region& region = checked_region(region_id, row);
    std::lock_guard<std::mutex> state_guard(state_mutex_);
    return region.resident[row] != 0;
  }

  py::dict stats(bool reset) {
    std::lock_guard<std::mutex> guard(state_mutex_);
    py::dict histogram;
    histogram["buckets_us"] = latency_buckets_us_;
    histogram["counts"] = latency_counts_;
    py::dict out;
    out["fills"] = fills_;
    out["fills_from_prefetch"] = fills_from_prefetch_;
    out["fault_driven"] = fault_driven_;
    out["evictions"] = evictions_;
    out["resident_bytes"] = resident_bytes_;
    out["fill_latency_histogram"] = histogram;
    if (reset) {
      fills_ = 0;
      fills_from_prefetch_ = 0;
      fault_driven_ = 0;
      evictions_ = 0;
      std::fill(latency_counts_.begin(), latency_counts_.end(), 0);
    }
    return out;
  }

  void raise_if_error() {
    std::lock_guard<std::mutex> guard(error_mutex_);
    if (!fatal_error_.empty()) {
      throw std::runtime_error("UFFD handler failed: " + fatal_error_);
    }
  }

 private:
  static uint64_t row_key(int region, int row) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(region)) << 32) |
           static_cast<uint32_t>(row);
  }

  Region& checked_region(int region_id, int row) {
    if (region_id < 0 || region_id >= static_cast<int>(regions_.size())) {
      throw std::out_of_range("UFFD region id is out of range");
    }
    Region& region = *regions_[region_id];
    if (row < 0 || row >= region.num_rows) {
      throw std::out_of_range("expert row id is out of range for UFFD bank");
    }
    return region;
  }

  void wake_handler() noexcept {
    if (wake_fd_ < 0) return;
    const uint64_t one = 1;
    const ssize_t ignored = write(wake_fd_, &one, sizeof(one));
    (void)ignored;
  }

  void set_fatal(const std::string& error) {
    std::lock_guard<std::mutex> guard(error_mutex_);
    if (fatal_error_.empty()) fatal_error_ = error;
  }

  void ensure_bounce(size_t bytes) {
    if (bounce_size_ >= bytes) return;
    void* replacement = nullptr;
    const int err = posix_memalign(&replacement, page_size_, bytes);
    if (err != 0) {
      throw std::runtime_error("posix_memalign UFFD row buffer: " +
                               std::string(std::strerror(err)));
    }
    if (bounce_) free(bounce_);
    bounce_ = replacement;
    bounce_size_ = bytes;
  }

  bool evict_one(const std::unordered_set<uint64_t>& protected_rows) {
    int victim_region = -1;
    int victim_row = -1;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (int region_id = 0; region_id < static_cast<int>(regions_.size()); ++region_id) {
      Region& region = *regions_[region_id];
      for (int row = 0; row < region.num_rows; ++row) {
        if (!region.resident[row] || protected_rows.count(row_key(region_id, row))) continue;
        if (region.last_use[row] < oldest) {
          oldest = region.last_use[row];
          victim_region = region_id;
          victim_row = row;
        }
      }
    }
    if (victim_region < 0) return false;
    Region& victim = *regions_[victim_region];
    void* address = reinterpret_cast<void*>(
        victim.start + static_cast<uintptr_t>(victim_row) * victim.row_bytes);
    if (madvise(address, victim.row_bytes, MADV_DONTNEED) < 0) {
      throw_errno("MADV_DONTNEED UFFD expert row");
    }
    victim.resident[victim_row] = 0;
    victim.last_use[victim_row] = 0;
    resident_bytes_ -= victim.row_bytes;
    ++evictions_;
    return true;
  }

  size_t fill_row(int region_id, int row, bool from_prefetch,
                  const std::unordered_set<uint64_t>& protected_rows) {
    std::lock_guard<std::mutex> regions_guard(regions_mutex_);
    std::lock_guard<std::mutex> state_guard(state_mutex_);
    Region& region = checked_region(region_id, row);
    if (region.resident[row]) {
      region.last_use[row] = ++clock_;
      return 0;
    }
    while (resident_bytes_ + region.row_bytes > budget_bytes_) {
      if (!evict_one(protected_rows)) {
        throw std::runtime_error(
            "UFFD pager budget cannot hold the routed expert-row working set; "
            "increase --moe-pager-budget-gib");
      }
    }

    ensure_bounce(region.row_bytes);
    const auto begin = std::chrono::steady_clock::now();
    ring_->read_exact(
        region.source_fd, bounce_, region.row_bytes,
        region.file_offset + static_cast<uint64_t>(row) * region.row_bytes);
    struct uffdio_copy copy {};
    copy.src = reinterpret_cast<uintptr_t>(bounce_);
    copy.dst = region.start + static_cast<uintptr_t>(row) * region.row_bytes;
    copy.len = region.row_bytes;
    copy.mode = 0;
    if (ioctl(uffd_, UFFDIO_COPY, &copy) < 0) throw_errno("UFFDIO_COPY expert row");
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin).count();

    region.resident[row] = 1;
    region.last_use[row] = ++clock_;
    resident_bytes_ += region.row_bytes;
    ++fills_;
    if (from_prefetch) ++fills_from_prefetch_;
    else ++fault_driven_;
    const auto bucket = std::upper_bound(
        latency_buckets_us_.begin(), latency_buckets_us_.end(),
        static_cast<uint64_t>(std::max<int64_t>(elapsed, 0)));
    ++latency_counts_[static_cast<size_t>(bucket - latency_buckets_us_.begin())];
    return region.row_bytes / page_size_;
  }

  void finish_request(const std::shared_ptr<Request>& request,
                      size_t pages, const std::string& error) {
    {
      std::lock_guard<std::mutex> guard(request->mutex);
      request->pages = pages;
      request->error = error;
      request->done = true;
    }
    request->cv.notify_one();
  }

  void process_requests() {
    uint64_t value;
    while (read(wake_fd_, &value, sizeof(value)) < 0 && errno == EINTR) {}
    std::deque<std::shared_ptr<Request>> pending;
    {
      std::lock_guard<std::mutex> guard(request_mutex_);
      pending.swap(requests_);
    }
    for (const auto& request : pending) {
      size_t pages = 0;
      try {
        raise_if_error();
        std::unordered_set<uint64_t> protected_rows;
        size_t working_set = 0;
        {
          std::lock_guard<std::mutex> regions_guard(regions_mutex_);
          for (const int region_id : request->regions) {
            for (const int row : request->rows) {
              Region& region = checked_region(region_id, row);
              if (protected_rows.insert(row_key(region_id, row)).second) {
                if (working_set > budget_bytes_ - region.row_bytes) {
                  throw std::runtime_error(
                      "routed UFFD expert rows exceed --moe-pager-budget-gib");
                }
                working_set += region.row_bytes;
              }
            }
          }
        }
        for (const int region_id : request->regions) {
          for (const int row : request->rows) {
            pages += fill_row(region_id, row, true, protected_rows);
          }
        }
        finish_request(request, pages, "");
      } catch (const std::exception& exc) {
        finish_request(request, 0, exc.what());
      }
    }
  }

  void resolve_failed_fault(uintptr_t address) noexcept {
    struct uffdio_zeropage zero {};
    zero.range.start = address & ~(static_cast<uintptr_t>(page_size_) - 1);
    zero.range.len = page_size_;
    zero.mode = 0;
    ioctl(uffd_, UFFDIO_ZEROPAGE, &zero);
  }

  void process_faults() {
    while (true) {
      struct uffd_msg message {};
      const ssize_t got = read(uffd_, &message, sizeof(message));
      if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
      if (got < 0 && errno == EINTR) continue;
      if (got < 0) throw_errno("read userfaultfd event");
      if (got == 0) throw std::runtime_error("userfaultfd closed unexpectedly");
      if (got != sizeof(message) || message.event != UFFD_EVENT_PAGEFAULT) {
        throw std::runtime_error("unexpected userfaultfd event");
      }
      const uintptr_t address = message.arg.pagefault.address;
      try {
        int region_id = -1;
        int row = -1;
        {
          std::lock_guard<std::mutex> regions_guard(regions_mutex_);
          for (int i = 0; i < static_cast<int>(regions_.size()); ++i) {
            const Region& region = *regions_[i];
            if (address >= region.start && address < region.start + region.nbytes) {
              region_id = i;
              row = static_cast<int>((address - region.start) / region.row_bytes);
              break;
            }
          }
        }
        if (region_id < 0) throw std::runtime_error("fault outside registered bank rows");
        const std::unordered_set<uint64_t> protected_rows{
            row_key(region_id, row)};
        fill_row(region_id, row, false, protected_rows);
      } catch (const std::exception& exc) {
        set_fatal(exc.what());
        resolve_failed_fault(address);
      }
    }
  }

  void handler_main() noexcept {
    struct pollfd fds[2] = {
        {wake_fd_, POLLIN, 0},
        {uffd_, POLLIN, 0},
    };
    while (!stop_.load(std::memory_order_acquire)) {
      int ready;
      do {
        ready = poll(fds, 2, -1);
      } while (ready < 0 && errno == EINTR);
      if (ready < 0) {
        set_fatal("poll: " + std::string(std::strerror(errno)));
        return;
      }
      if (stop_.load(std::memory_order_acquire)) return;
      try {
        if (fds[0].revents & POLLIN) process_requests();
        if (fds[1].revents & POLLIN) process_faults();
      } catch (const std::exception& exc) {
        set_fatal(exc.what());
      }
    }
  }

  uint64_t budget_bytes_;
  size_t page_size_;
  int uffd_ = -1;
  int wake_fd_ = -1;
  std::unique_ptr<IoUring> ring_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::mutex regions_mutex_;
  std::vector<std::unique_ptr<Region>> regions_;
  std::mutex request_mutex_;
  std::deque<std::shared_ptr<Request>> requests_;
  std::mutex state_mutex_;
  uint64_t clock_ = 0;
  uint64_t fills_ = 0;
  uint64_t fills_from_prefetch_ = 0;
  uint64_t fault_driven_ = 0;
  uint64_t evictions_ = 0;
  uint64_t resident_bytes_ = 0;
  const std::vector<uint64_t> latency_buckets_us_{
      50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000};
  std::vector<uint64_t> latency_counts_ =
      std::vector<uint64_t>(latency_buckets_us_.size() + 1, 0);
  void* bounce_ = nullptr;
  size_t bounce_size_ = 0;
  std::mutex error_mutex_;
  std::string fatal_error_;
};

}  // namespace

#endif  // __linux__

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef __linux__
  namespace py = pybind11;
  m.def("probe", &probe);
  py::class_<UffdPager>(m, "UffdPager")
      .def(py::init<uint64_t>(), py::arg("budget_bytes"))
      .def("add_region", &UffdPager::add_region,
           py::arg("address"), py::arg("length"), py::arg("nbytes"),
           py::arg("path"), py::arg("file_offset"), py::arg("row_bytes"),
           py::arg("num_rows"))
      .def("prefetch", &UffdPager::prefetch,
           py::arg("regions"), py::arg("rows"),
           py::call_guard<py::gil_scoped_release>())
      .def("is_resident", &UffdPager::is_resident,
           py::arg("region"), py::arg("row"))
      .def("stats", &UffdPager::stats, py::arg("reset") = false)
      .def("raise_if_error", &UffdPager::raise_if_error);
#else
  m.def("probe", []() {
    throw std::runtime_error("userfaultfd pager is Linux-only");
  });
#endif
}
