// Thin Metal runtime: device, run-time kernel compilation with caching, buffers.
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <map>
#include <string>

namespace h3 {

// Preprocessor macros handed to the Metal compiler; they select tile sizes,
// element types and other per-pipeline specializations.
using Macros = std::map<std::string, std::string>;

class Metal {
 public:
  explicit Metal(const std::string &kernel_dir);

  id<MTLDevice> device() const { return dev_; }
  id<MTLCommandQueue> queue() const { return queue_; }

  // Compiles kernels/<file> (resolving `#include "..."` against the kernel
  // directory) and returns the compute pipeline for `fn`. Results are cached by
  // (file, fn, macros). On failure returns nil and fills *err.
  id<MTLComputePipelineState> pipeline(const std::string &file, const std::string &fn, const Macros &macros,
                                       std::string *err);

  // Shared-storage buffer, zero-filled, length rounded up to a whole page.
  id<MTLBuffer> alloc(size_t bytes);

  // Wraps existing page-aligned memory (e.g. an mmap'd weight file) without copying.
  id<MTLBuffer> wrap(void *ptr, size_t bytes);

  static size_t page_size();
  static size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

 private:
  bool load_source(const std::string &file, std::string *out, std::string *err, int depth = 0);

  id<MTLDevice> dev_;
  id<MTLCommandQueue> queue_;
  std::string kernel_dir_;
  std::map<std::string, id<MTLComputePipelineState>> cache_;
};

}  // namespace h3
