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
#include <sys/stat.h>
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
#include <set>
#include <string>
#include <thread>
#include <tuple>
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

void pread_exact(int source_fd, void* dst, size_t length, uint64_t offset) {
  size_t completed = 0;
  while (completed < length) {
    ssize_t got;
    do {
      got = pread(
          source_fd, static_cast<char*>(dst) + completed, length - completed,
          static_cast<off_t>(offset + completed));
    } while (got < 0 && errno == EINTR);
    if (got < 0) throw_errno("pread FTW page");
    if (got == 0) {
      throw std::runtime_error(
          "unexpected EOF during pread FTW page: got " +
          std::to_string(completed) + ", expected " + std::to_string(length));
    }
    completed += static_cast<size_t>(got);
  }
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
      throw std::runtime_error("UFFD page is too large for one io_uring read");
    }
    size_t completed = 0;
    while (completed < length) {
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
      sqe->off = offset + completed;
      sqe->addr = reinterpret_cast<uint64_t>(
          static_cast<char*>(dst) + completed);
      sqe->len = static_cast<unsigned>(length - completed);
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
        throw_errno("io_uring FTW page read");
      }
      if (cqe.res == 0) {
        throw std::runtime_error(
            "unexpected EOF during io_uring FTW page read: got " +
            std::to_string(completed) + ", expected " + std::to_string(length));
      }
      completed += static_cast<size_t>(cqe.res);
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
  std::vector<uint64_t> page_last_use;

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
    if (address % page_size_ || length == 0 || length % page_size_) {
      throw std::invalid_argument(
          "UFFD bank address and mapping length must be page-aligned");
    }
    if (num_rows <= 0 || row_bytes == 0 ||
        row_bytes > std::numeric_limits<size_t>::max() / static_cast<size_t>(num_rows) ||
        row_bytes * static_cast<size_t>(num_rows) != nbytes || nbytes > length ||
        length - nbytes >= page_size_) {
      throw std::invalid_argument(
          "UFFD bank mapping must contain equal expert rows plus at most one "
          "partial page of mapping padding");
    }
    if (page_size_ > budget_bytes_) {
      throw std::invalid_argument(
          "UFFD pager budget is smaller than one system page");
    }
    const int source_fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (source_fd < 0) throw_errno("open FTW bank source " + path);
    struct stat source_stat {};
    if (fstat(source_fd, &source_stat) < 0) {
      const int saved = errno;
      close(source_fd);
      errno = saved;
      throw_errno("fstat FTW bank source " + path);
    }
    if (source_stat.st_size < 0 ||
        file_offset > std::numeric_limits<uint64_t>::max() - nbytes ||
        file_offset + nbytes > static_cast<uint64_t>(source_stat.st_size)) {
      close(source_fd);
      throw std::runtime_error("UFFD FTW bank source range exceeds file " + path);
    }

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
    const size_t num_pages = length / page_size_;
    region->page_last_use.assign(num_pages, 0);
    uint64_t spanning_rows = 0;
    for (int row = 0; row < num_rows; ++row) {
      const size_t first = static_cast<size_t>(row) * row_bytes / page_size_;
      const size_t end = (static_cast<size_t>(row + 1) * row_bytes +
                          page_size_ - 1) / page_size_;
      if (end - first > 1) ++spanning_rows;
    }
    std::lock_guard<std::mutex> guard(regions_mutex_);
    const int id = static_cast<int>(regions_.size());
    try {
      regions_.push_back(std::move(region));
    } catch (...) {
      struct uffdio_range range {address, length};
      ioctl(uffd_, UFFDIO_UNREGISTER, &range);
      throw;
    }
    {
      std::lock_guard<std::mutex> state_guard(state_mutex_);
      rows_spanning_pages_ += spanning_rows;
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
    const size_t first = row_first_page(region, row);
    const size_t end = row_page_end(region, row);
    return std::all_of(
        region.page_last_use.begin() + first,
        region.page_last_use.begin() + end,
        [](uint64_t last_use) { return last_use != 0; });
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
    out["pages_installed"] = pages_installed_;
    out["rows_spanning_pages"] = rows_spanning_pages_;
    out["fill_latency_histogram"] = histogram;
    if (reset) {
      fills_ = 0;
      fills_from_prefetch_ = 0;
      fault_driven_ = 0;
      evictions_ = 0;
      pages_installed_ = 0;
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
  uint64_t page_key(const Region& region, size_t page) const {
    return static_cast<uint64_t>(region.start) + page * page_size_;
  }

  size_t row_first_page(const Region& region, int row) const {
    return static_cast<size_t>(row) * region.row_bytes / page_size_;
  }

  size_t row_page_end(const Region& region, int row) const {
    return (static_cast<size_t>(row + 1) * region.row_bytes + page_size_ - 1) /
           page_size_;
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
      throw std::runtime_error("posix_memalign UFFD page buffer: " +
                               std::string(std::strerror(err)));
    }
    if (bounce_) free(bounce_);
    bounce_ = replacement;
    bounce_size_ = bytes;
  }

  bool evict_one(const std::unordered_set<uint64_t>& protected_pages) {
    auto victim = page_lru_.begin();
    while (victim != page_lru_.end()) {
      const auto& [last_use, region_id, page] = *victim;
      (void)last_use;
      if (!protected_pages.count(page_key(*regions_[region_id], page))) break;
      ++victim;
    }
    if (victim == page_lru_.end()) return false;
    const auto [last_use, victim_region, victim_page] = *victim;
    (void)last_use;
    Region& region = *regions_[victim_region];
    void* address = reinterpret_cast<void*>(
        region.start + static_cast<uintptr_t>(victim_page) * page_size_);
    if (madvise(address, page_size_, MADV_DONTNEED) < 0) {
      throw_errno("MADV_DONTNEED UFFD bank page");
    }
    region.page_last_use[victim_page] = 0;
    page_lru_.erase(victim);
    resident_bytes_ -= page_size_;
    ++evictions_;
    return true;
  }

  void touch_page(Region& region, int region_id, size_t page) {
    const uint64_t previous = region.page_last_use[page];
    if (previous) {
      page_lru_.erase(std::make_tuple(previous, region_id, page));
    }
    const uint64_t current = ++clock_;
    region.page_last_use[page] = current;
    page_lru_.emplace(current, region_id, page);
  }

  size_t fill_page(int region_id, size_t page,
                   const std::unordered_set<uint64_t>& protected_pages) {
    std::lock_guard<std::mutex> regions_guard(regions_mutex_);
    std::lock_guard<std::mutex> state_guard(state_mutex_);
    if (region_id < 0 || region_id >= static_cast<int>(regions_.size())) {
      throw std::out_of_range("UFFD region id is out of range");
    }
    Region& region = *regions_[region_id];
    if (page >= region.page_last_use.size()) {
      throw std::out_of_range("UFFD page id is out of range for bank");
    }
    if (region.page_last_use[page]) {
      touch_page(region, region_id, page);
      return 0;
    }
    while (resident_bytes_ + page_size_ > budget_bytes_) {
      if (!evict_one(protected_pages)) {
        throw std::runtime_error(
            "UFFD pager budget cannot hold the routed expert-page working set; "
            "increase --moe-pager-budget-gib");
      }
    }

    ensure_bounce(page_size_);
    std::memset(bounce_, 0, page_size_);
    const size_t bank_offset = page * page_size_;
    const size_t valid_bytes = bank_offset < region.nbytes
        ? std::min(page_size_, region.nbytes - bank_offset)
        : 0;
    if (valid_bytes) {
      const uint64_t source_offset =
          region.file_offset + static_cast<uint64_t>(bank_offset);
      if (source_offset % page_size_ == 0 && valid_bytes == page_size_) {
        ring_->read_exact(region.source_fd, bounce_, valid_bytes, source_offset);
      } else {
        pread_exact(region.source_fd, bounce_, valid_bytes, source_offset);
      }
    }
    struct uffdio_copy copy {};
    copy.src = reinterpret_cast<uintptr_t>(bounce_);
    copy.dst = region.start + static_cast<uintptr_t>(page) * page_size_;
    copy.len = page_size_;
    copy.mode = 0;
    if (ioctl(uffd_, UFFDIO_COPY, &copy) < 0) throw_errno("UFFDIO_COPY bank page");
    touch_page(region, region_id, page);
    resident_bytes_ += page_size_;
    ++pages_installed_;
    return 1;
  }

  void record_fill(bool from_prefetch,
                   std::chrono::steady_clock::duration duration) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        duration).count();
    std::lock_guard<std::mutex> state_guard(state_mutex_);
    ++fills_;
    if (from_prefetch) ++fills_from_prefetch_;
    else ++fault_driven_;
    const auto bucket = std::upper_bound(
        latency_buckets_us_.begin(), latency_buckets_us_.end(),
        static_cast<uint64_t>(std::max<int64_t>(elapsed, 0)));
    ++latency_counts_[static_cast<size_t>(bucket - latency_buckets_us_.begin())];
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
        std::unordered_set<uint64_t> protected_pages;
        size_t working_set = 0;
        {
          std::lock_guard<std::mutex> regions_guard(regions_mutex_);
          for (const int region_id : request->regions) {
            for (const int row : request->rows) {
              Region& region = checked_region(region_id, row);
              for (size_t page = row_first_page(region, row);
                   page < row_page_end(region, row); ++page) {
                if (!protected_pages.insert(page_key(region, page)).second) continue;
                if (working_set > budget_bytes_ - page_size_) {
                  throw std::runtime_error(
                      "routed UFFD expert pages exceed --moe-pager-budget-gib");
                }
                working_set += page_size_;
              }
            }
          }
        }
        for (const int region_id : request->regions) {
          for (const int row : request->rows) {
            Region* region;
            {
              std::lock_guard<std::mutex> regions_guard(regions_mutex_);
              region = &checked_region(region_id, row);
            }
            const auto begin = std::chrono::steady_clock::now();
            size_t row_pages = 0;
            for (size_t page = row_first_page(*region, row);
                 page < row_page_end(*region, row); ++page) {
              row_pages += fill_page(region_id, page, protected_pages);
            }
            pages += row_pages;
            if (row_pages) {
              record_fill(true, std::chrono::steady_clock::now() - begin);
            }
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
        size_t page = 0;
        {
          std::lock_guard<std::mutex> regions_guard(regions_mutex_);
          for (int i = 0; i < static_cast<int>(regions_.size()); ++i) {
            const Region& region = *regions_[i];
            if (address >= region.start && address < region.start + region.length) {
              region_id = i;
              page = (address - region.start) / page_size_;
              break;
            }
          }
        }
        if (region_id < 0) throw std::runtime_error("fault outside registered bank mapping");
        uint64_t key;
        {
          std::lock_guard<std::mutex> regions_guard(regions_mutex_);
          key = page_key(*regions_[region_id], page);
        }
        const std::unordered_set<uint64_t> protected_pages{key};
        const auto begin = std::chrono::steady_clock::now();
        if (fill_page(region_id, page, protected_pages)) {
          record_fill(false, std::chrono::steady_clock::now() - begin);
        }
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
  std::set<std::tuple<uint64_t, int, size_t>> page_lru_;
  uint64_t fills_ = 0;
  uint64_t fills_from_prefetch_ = 0;
  uint64_t fault_driven_ = 0;
  uint64_t evictions_ = 0;
  uint64_t resident_bytes_ = 0;
  uint64_t pages_installed_ = 0;
  uint64_t rows_spanning_pages_ = 0;
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
