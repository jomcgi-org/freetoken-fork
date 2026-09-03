// Strict io_uring PLE row reader. Linux only, single issuer, no liburing dependency.
// Rows are read through page-aligned bounce slots and copied into caller-owned pinned
// staging. O_DIRECT is preferred per file; buffered io_uring is the explicit fallback.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <linux/io_uring.h>
#include <linux/time_types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <torch/extension.h>

namespace py = pybind11;

namespace {

constexpr int64_t kDirectAlignment = 4096;
constexpr int64_t kBounceSpan = 2 * kDirectAlignment;
constexpr unsigned kMaxQueueDepth = 4096;
// This timeout is the wait granularity. The total read and drain bounds below
// control how long repeated waits may continue.
constexpr int64_t kEnterTimeoutSeconds = 1;
constexpr int64_t kReadWallTimeoutSeconds = 60;
// Error recovery gets a longer chance to make the store reusable.
constexpr int64_t kDrainPatienceSeconds = 5;
// Teardown bounds per-store latency before preserving an in-flight buffer.
constexpr int64_t kTeardownDrainPatienceSeconds = 1;

std::string errno_text(const char *operation, int error) {
  return std::string(operation) + ": " + std::strerror(error) + " (errno " +
         std::to_string(error) + ")";
}

uint8_t *aligned_alloc_or_throw(size_t bytes) {
  void *ptr = nullptr;
  const int rc = posix_memalign(&ptr, kDirectAlignment, bytes);
  if (rc != 0) throw std::runtime_error(errno_text("posix_memalign", rc));
  return static_cast<uint8_t *>(ptr);
}

class TableFile {
 public:
  explicit TableFile(std::string path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    direct_ = fd_ >= 0;
    direct_error_ = direct_ ? 0 : errno;
    if (fd_ < 0) fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error(path_ + ": " + std::strerror(errno));

    struct stat status {};
    if (::fstat(fd_, &status) != 0) {
      const int error = errno;
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error(path_ + ": " + errno_text("fstat", error));
    }
    size_ = status.st_size;
    if (!direct_) (void)::posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
  }

  ~TableFile() {
    if (fd_ >= 0) ::close(fd_);
  }

  TableFile(const TableFile &) = delete;
  TableFile &operator=(const TableFile &) = delete;

  int fd() const { return fd_; }
  int64_t size() const { return size_; }
  bool direct() const { return direct_; }
  int direct_error() const { return direct_error_; }
  const std::string &path() const { return path_; }

  void discard_cache(int64_t offset, int64_t length) const {
    if (!direct_) (void)::posix_fadvise(fd_, offset, length, POSIX_FADV_DONTNEED);
  }

 private:
  std::string path_;
  int fd_ = -1;
  int64_t size_ = 0;
  bool direct_ = false;
  int direct_error_ = 0;
};

// uffd_pager.cpp:98-252 has a separate io_uring implementation. A shared
// ring setup and wait header is a later refactor.
class IoUring {
  enum class EnterResult { kSuccess, kRetry, kFailure };

 public:
  explicit IoUring(unsigned requested_entries) {
    if (requested_entries < 1 || requested_entries > kMaxQueueDepth) {
      throw std::runtime_error("io_uring queue depth must be in [1, 4096]");
    }
    io_uring_params params {};
    fd_ = static_cast<int>(::syscall(__NR_io_uring_setup, requested_entries, &params));
    if (fd_ < 0) {
      const int error = errno;
      throw std::runtime_error(
          "io_uring unavailable: io_uring_setup(queue_depth=" +
          std::to_string(requested_entries) + ") failed: " +
          std::strerror(error) + " (errno " + std::to_string(error) + ")"
      );
    }
    if (!(params.features & IORING_FEAT_EXT_ARG)) {
      cleanup();
      throw std::runtime_error(
          "io_uring unavailable: IORING_FEAT_EXT_ARG requires Linux kernel "
          "5.11+");
    }

    sq_size_ = params.sq_off.array + params.sq_entries * sizeof(uint32_t);
    cq_size_ = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
    if (params.features & IORING_FEAT_SINGLE_MMAP) {
      sq_size_ = cq_size_ = std::max(sq_size_, cq_size_);
    }
    sq_ptr_ = ::mmap(nullptr, sq_size_, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQ_RING);
    if (sq_ptr_ == MAP_FAILED) throw_map("mmap io_uring submission ring");
    cq_ptr_ = (params.features & IORING_FEAT_SINGLE_MMAP)
                  ? sq_ptr_
                  : ::mmap(nullptr, cq_size_, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_CQ_RING);
    if (cq_ptr_ == MAP_FAILED) throw_map("mmap io_uring completion ring");
    sqes_size_ = params.sq_entries * sizeof(io_uring_sqe);
    sqes_ = static_cast<io_uring_sqe *>(
        ::mmap(nullptr, sqes_size_, PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQES));
    if (sqes_ == MAP_FAILED) throw_map("mmap io_uring SQEs");

    auto at = [](void *base, uint32_t offset) {
      return static_cast<uint8_t *>(base) + offset;
    };
    sq_tail_ = reinterpret_cast<uint32_t *>(at(sq_ptr_, params.sq_off.tail));
    sq_mask_ = reinterpret_cast<uint32_t *>(at(sq_ptr_, params.sq_off.ring_mask));
    sq_array_ = reinterpret_cast<uint32_t *>(at(sq_ptr_, params.sq_off.array));
    cq_head_ = reinterpret_cast<uint32_t *>(at(cq_ptr_, params.cq_off.head));
    cq_tail_ = reinterpret_cast<uint32_t *>(at(cq_ptr_, params.cq_off.tail));
    cq_mask_ = reinterpret_cast<uint32_t *>(at(cq_ptr_, params.cq_off.ring_mask));
    cqes_ = reinterpret_cast<io_uring_cqe *>(at(cq_ptr_, params.cq_off.cqes));
    entries_ = params.sq_entries;
    required_.assign(entries_, 0);
    sq_shadow_tail_ = __atomic_load_n(sq_tail_, __ATOMIC_ACQUIRE);
  }

  ~IoUring() {
    (void)shutdown();
  }

  IoUring(const IoUring &) = delete;
  IoUring &operator=(const IoUring &) = delete;

  unsigned capacity() const { return entries_; }

  bool shutdown() noexcept {
    if (closed_) return drained_;
    stop_requested_.store(true, std::memory_order_release);
    drained_ = drain(kTeardownDrainPatienceSeconds);
    cleanup();
    closed_ = true;
    return drained_;
  }

  void submit(unsigned tag, int file_fd, uint8_t *buffer, int64_t length,
              int64_t required, int64_t offset) {
    io_uring_sqe *sqe = &sqes_[sq_shadow_tail_ & *sq_mask_];
    std::memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ;
    sqe->fd = file_fd;
    sqe->addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer));
    sqe->len = static_cast<uint32_t>(length);
    sqe->off = static_cast<uint64_t>(offset);
    sqe->user_data = tag;
    sq_array_[sq_shadow_tail_ & *sq_mask_] = sq_shadow_tail_ & *sq_mask_;
    ++sq_shadow_tail_;
    __atomic_store_n(sq_tail_, sq_shadow_tail_, __ATOMIC_RELEASE);
    required_.at(tag) = required;
    ++to_submit_;
    ++in_flight_;
  }

  unsigned wait_one(
      const std::chrono::steady_clock::time_point &read_deadline) {
    for (;;) {
      if (stop_requested_.load(std::memory_order_acquire)) {
        throw std::runtime_error("io_uring read stopped during shutdown");
      }
      if (std::chrono::steady_clock::now() >= read_deadline) {
        throw std::runtime_error(
            "io_uring read timed out after " +
            std::to_string(kReadWallTimeoutSeconds) + " seconds");
      }
      const uint32_t head = __atomic_load_n(cq_head_, __ATOMIC_RELAXED);
      const uint32_t tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
      if (head != tail) {
        const io_uring_cqe &completion = cqes_[head & *cq_mask_];
        const unsigned tag = static_cast<unsigned>(completion.user_data);
        const int result = completion.res;
        __atomic_store_n(cq_head_, head + 1, __ATOMIC_RELEASE);
        --in_flight_;
        if (result < 0) {
          throw std::runtime_error(
              errno_text("io_uring read", -result));
        }
        if (static_cast<int64_t>(result) < required_.at(tag)) {
          throw std::runtime_error(
              "io_uring short read: got " + std::to_string(result) +
              " bytes, need " + std::to_string(required_.at(tag)));
        }
        return tag;
      }
      if (enter(/*minimum_completions=*/1, /*throw_errors=*/true) ==
          EnterResult::kRetry) {
        throw_if_python_signal_pending();
      }
    }
  }

  bool drain(int64_t patience_seconds = kDrainPatienceSeconds) noexcept {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(patience_seconds);
    while (in_flight_ > 0) {
      const uint32_t head = __atomic_load_n(cq_head_, __ATOMIC_RELAXED);
      const uint32_t tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
      if (head != tail) {
        __atomic_store_n(cq_head_, head + 1, __ATOMIC_RELEASE);
        --in_flight_;
      } else {
        if (std::chrono::steady_clock::now() >= deadline ||
            enter(/*minimum_completions=*/1, /*throw_errors=*/false) ==
                EnterResult::kFailure) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  [[noreturn]] void throw_map(const char *operation) {
    const int error = errno;
    cleanup();
    throw std::runtime_error(
        std::string("io_uring unavailable: ") + errno_text(operation, error));
  }

  void cleanup() noexcept {
    if (sqes_ != nullptr && sqes_ != MAP_FAILED) ::munmap(sqes_, sqes_size_);
    if (cq_ptr_ != nullptr && cq_ptr_ != MAP_FAILED && cq_ptr_ != sq_ptr_) {
      ::munmap(cq_ptr_, cq_size_);
    }
    if (sq_ptr_ != nullptr && sq_ptr_ != MAP_FAILED) ::munmap(sq_ptr_, sq_size_);
    if (fd_ >= 0) ::close(fd_);
    sqes_ = nullptr;
    cq_ptr_ = nullptr;
    sq_ptr_ = nullptr;
    fd_ = -1;
  }

  EnterResult enter(unsigned minimum_completions, bool throw_errors) {
    const unsigned submitting = to_submit_;
    __kernel_timespec timeout {};
    timeout.tv_sec = kEnterTimeoutSeconds;
    io_uring_getevents_arg arguments {};
    arguments.ts = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(&timeout));
    const long result = ::syscall(
        __NR_io_uring_enter, fd_, submitting, minimum_completions,
        IORING_ENTER_GETEVENTS | IORING_ENTER_EXT_ARG, &arguments,
        sizeof(arguments));
    if (result >= 0) {
      to_submit_ = submitting - static_cast<unsigned>(result);
      return EnterResult::kSuccess;
    }
    const int error = errno;
    if (error == EINTR || error == ETIME) return EnterResult::kRetry;
    if (throw_errors) {
      throw std::runtime_error(
          "io_uring unavailable during submission: " +
          errno_text("io_uring_enter", error));
    }
    return EnterResult::kFailure;
  }

  void throw_if_python_signal_pending() {
    py::gil_scoped_acquire acquire;
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
  }

  int fd_ = -1;
  void *sq_ptr_ = nullptr;
  void *cq_ptr_ = nullptr;
  io_uring_sqe *sqes_ = nullptr;
  size_t sq_size_ = 0;
  size_t cq_size_ = 0;
  size_t sqes_size_ = 0;
  uint32_t *sq_tail_ = nullptr;
  uint32_t *sq_mask_ = nullptr;
  uint32_t *sq_array_ = nullptr;
  uint32_t *cq_head_ = nullptr;
  uint32_t *cq_tail_ = nullptr;
  uint32_t *cq_mask_ = nullptr;
  io_uring_cqe *cqes_ = nullptr;
  unsigned entries_ = 0;
  uint32_t sq_shadow_tail_ = 0;
  unsigned to_submit_ = 0;
  unsigned in_flight_ = 0;
  std::vector<int64_t> required_;
  // Dormant for interrupting an active pybind read until an explicit close API
  // exists: shutdown is currently reachable only during destruction.
  std::atomic<bool> stop_requested_{false};
  bool closed_ = false;
  bool drained_ = true;
};

class UringRowStore {
  struct Extent {
    const TableFile *file;
    int64_t base;
  };

  struct Pending {
    const TableFile *file;
    int64_t read_offset;
    int64_t read_length;
    int64_t row_offset;
    std::vector<uint8_t *> destinations;
  };

 public:
  UringRowStore(std::vector<std::string> paths,
                std::vector<int64_t> extent_file,
                std::vector<int64_t> extent_base,
                int64_t rows_per_extent, int64_t row_bytes,
                int64_t row_stride, unsigned queue_depth)
      : rows_per_extent_(rows_per_extent),
        row_bytes_(row_bytes),
        row_stride_(row_stride),
        ring_(std::make_unique<IoUring>(queue_depth)) {
    if (paths.empty() || extent_file.empty() ||
        extent_file.size() != extent_base.size()) {
      throw std::runtime_error(
          "PLE uring source needs files and equal nonempty extent vectors");
    }
    if (rows_per_extent_ < 1 || row_bytes_ < 1 || row_stride_ < row_bytes_) {
      throw std::runtime_error(
          "PLE uring geometry requires positive rows and row_stride >= row_bytes");
    }
    if (row_bytes_ > kDirectAlignment) {
      throw std::runtime_error(
          "PLE uring row_bytes " + std::to_string(row_bytes_) +
          " exceeds the 4096-byte bounce-row limit");
    }
    for (std::string &path : paths) {
      files_.push_back(std::make_unique<TableFile>(std::move(path)));
    }
    const int64_t extent_bytes =
        (rows_per_extent_ - 1) * row_stride_ + row_bytes_;
    for (size_t index = 0; index < extent_file.size(); ++index) {
      const int64_t file_index = extent_file[index];
      const int64_t base = extent_base[index];
      if (file_index < 0 || static_cast<size_t>(file_index) >= files_.size()) {
        throw std::runtime_error("PLE uring extent has an invalid file index");
      }
      if (base < 0 || base + extent_bytes > files_[file_index]->size()) {
        throw std::runtime_error(
            files_[file_index]->path() + ": PLE extent needs " +
            std::to_string(base + extent_bytes) + " bytes, file has " +
            std::to_string(files_[file_index]->size()));
      }
      extents_.push_back(Extent{files_[file_index].get(), base});
    }
    bounce_ = aligned_alloc_or_throw(
        static_cast<size_t>(ring_->capacity()) * kBounceSpan);
  }

  ~UringRowStore() {
    const size_t bounce_bytes =
        static_cast<size_t>(ring_->capacity()) * kBounceSpan;
    bool drained;
    {
      py::gil_scoped_release release;
      drained = ring_->shutdown();
    }
    ring_.reset();
    if (drained) {
      std::free(bounce_);
    } else {
      std::fprintf(
          stderr,
          "PLE uring drain failed; leaking %zu-byte bounce buffer to preserve "
          "in-flight read lifetime\n",
          bounce_bytes);
    }
    bounce_ = nullptr;
  }

  UringRowStore(const UringRowStore &) = delete;
  UringRowStore &operator=(const UringRowStore &) = delete;

  size_t read_rows(uintptr_t row_ids_address, int64_t count,
                   uintptr_t destination_address, int64_t destination_stride) {
    if (poisoned_) {
      throw std::runtime_error(
          "PLE uring row store is poisoned after an incomplete drain");
    }
    if (count < 0 || destination_stride < row_bytes_) {
      throw std::runtime_error("PLE uring read_rows received invalid geometry");
    }
    const int64_t *row_ids = reinterpret_cast<const int64_t *>(row_ids_address);
    uint8_t *destination = reinterpret_cast<uint8_t *>(destination_address);
    std::vector<Pending> pending;
    std::unordered_map<int64_t, size_t> pending_index;
    pending.reserve(static_cast<size_t>(count));
    pending_index.reserve(static_cast<size_t>(count));
    const int64_t total_rows =
        static_cast<int64_t>(extents_.size()) * rows_per_extent_;

    for (int64_t index = 0; index < count; ++index) {
      const int64_t row_id = row_ids[index];
      if (row_id < 0 || row_id >= total_rows) {
        throw std::runtime_error(
            "PLE uring row id " + std::to_string(row_id) +
            " is outside [0, " + std::to_string(total_rows) + ")");
      }
      uint8_t *row_destination = destination + index * destination_stride;
      const auto found = pending_index.find(row_id);
      if (found != pending_index.end()) {
        pending[found->second].destinations.push_back(row_destination);
        continue;
      }
      const Extent &extent = extents_[row_id / rows_per_extent_];
      const int64_t offset =
          extent.base + (row_id % rows_per_extent_) * row_stride_;
      Pending request{extent.file, offset, row_bytes_, 0, {row_destination}};
      if (extent.file->direct()) {
        request.read_offset = offset & ~(kDirectAlignment - 1);
        request.row_offset = offset - request.read_offset;
        request.read_length =
            ((offset + row_bytes_ + kDirectAlignment - 1) &
             ~(kDirectAlignment - 1)) -
            request.read_offset;
      }
      pending_index.emplace(row_id, pending.size());
      pending.push_back(std::move(request));
    }
    flush(pending);
    return pending.size();
  }

  unsigned queue_depth() const { return ring_->capacity(); }
  size_t file_count() const { return files_.size(); }

  size_t direct_file_count() const {
    return static_cast<size_t>(std::count_if(
        files_.begin(), files_.end(),
        [](const std::unique_ptr<TableFile> &file) { return file->direct(); }));
  }

  std::vector<std::string> direct_fallbacks() const {
    std::vector<std::string> result;
    for (const auto &file : files_) {
      if (!file->direct()) {
        result.push_back(file->path() + ": " + std::strerror(file->direct_error()));
      }
    }
    return result;
  }

 private:
  void flush(const std::vector<Pending> &pending) {
    if (pending.empty()) return;
    const unsigned capacity = ring_->capacity();
    std::vector<size_t> tag_to_pending(capacity);
    size_t next = 0;
    auto submit = [&](unsigned tag) {
      const Pending &request = pending[next];
      tag_to_pending[tag] = next++;
      ring_->submit(tag, request.file->fd(),
                    bounce_ + static_cast<size_t>(tag) * kBounceSpan,
                    request.read_length,
                    request.row_offset + row_bytes_, request.read_offset);
    };
    try {
      const size_t initial = std::min(static_cast<size_t>(capacity), pending.size());
      for (unsigned tag = 0; tag < initial; ++tag) submit(tag);
      auto read_deadline = std::chrono::steady_clock::now() +
                           std::chrono::seconds(kReadWallTimeoutSeconds);
      for (size_t completed = 0; completed < pending.size(); ++completed) {
        const unsigned tag = ring_->wait_one(read_deadline);
        read_deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(kReadWallTimeoutSeconds);
        const Pending &request = pending[tag_to_pending[tag]];
        const uint8_t *row =
            bounce_ + static_cast<size_t>(tag) * kBounceSpan + request.row_offset;
        for (uint8_t *destination : request.destinations) {
          std::memcpy(destination, row, static_cast<size_t>(row_bytes_));
        }
        request.file->discard_cache(request.read_offset, request.read_length);
        if (next < pending.size()) submit(tag);
      }
    } catch (...) {
      if (!ring_->drain()) poisoned_ = true;
      throw;
    }
  }

  int64_t rows_per_extent_;
  int64_t row_bytes_;
  int64_t row_stride_;
  std::vector<std::unique_ptr<TableFile>> files_;
  std::vector<Extent> extents_;
  // The explicit destructor tears down the ring before its DMA target and files.
  uint8_t *bounce_ = nullptr;
  std::unique_ptr<IoUring> ring_;
  bool poisoned_ = false;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<UringRowStore>(module, "UringRowStore")
      .def(py::init<std::vector<std::string>, std::vector<int64_t>,
                    std::vector<int64_t>, int64_t, int64_t, int64_t, unsigned>(),
           py::arg("paths"), py::arg("extent_file"), py::arg("extent_base"),
           py::arg("rows_per_extent"), py::arg("row_bytes"),
           py::arg("row_stride"), py::arg("queue_depth"))
      .def("read_rows", &UringRowStore::read_rows, py::arg("row_ids_address"),
           py::arg("count"), py::arg("destination_address"),
           py::arg("destination_stride"), py::call_guard<py::gil_scoped_release>())
      .def("queue_depth", &UringRowStore::queue_depth)
      .def("file_count", &UringRowStore::file_count)
      .def("direct_file_count", &UringRowStore::direct_file_count)
      .def("direct_fallbacks", &UringRowStore::direct_fallbacks);
}
