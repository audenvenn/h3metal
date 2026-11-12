#include "metal_rt.h"

#include <fstream>
#include <sstream>
#include <unistd.h>

namespace h3 {

size_t Metal::page_size() { return (size_t)getpagesize(); }

Metal::Metal(const std::string &kernel_dir) : kernel_dir_(kernel_dir) {
  dev_ = MTLCreateSystemDefaultDevice();
  queue_ = dev_ ? [dev_ newCommandQueue] : nil;
}

bool Metal::load_source(const std::string &file, std::string *out, std::string *err, int depth) {
  if (depth > 8) {
    *err = "kernel include depth exceeded at " + file;
    return false;
  }
  std::ifstream in(kernel_dir_ + "/" + file);
  if (!in) {
    *err = "cannot read kernel source " + kernel_dir_ + "/" + file;
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    // Run-time compilation has no include path, so local includes are spliced in by hand.
    size_t pos = line.find("#include \"");
    if (pos != std::string::npos && line.find_first_not_of(" \t") == pos) {
      size_t a = pos + 10, b = line.find('"', a);
      if (!load_source(line.substr(a, b - a), out, err, depth + 1)) return false;
      *out += "\n";
    } else {
      *out += line + "\n";
    }
  }
  return true;
}

id<MTLComputePipelineState> Metal::pipeline(const std::string &file, const std::string &fn, const Macros &macros,
                                            std::string *err) {
  std::string key = file + "|" + fn;
  for (const auto &[k, v] : macros) key += "|" + k + "=" + v;
  auto it = cache_.find(key);
  if (it != cache_.end()) return it->second;

  std::string src;
  if (!load_source(file, &src, err)) return nil;

  MTLCompileOptions *opt = [MTLCompileOptions new];
  opt.languageVersion = MTLLanguageVersion4_1;
  // No value-changing floating-point rewrites: results must not depend on how
  // the compiler felt about reassociating a sum.
  opt.mathMode = MTLMathModeSafe;
  NSMutableDictionary<NSString *, NSObject *> *defs = [NSMutableDictionary dictionary];
  for (const auto &[k, v] : macros) defs[@(k.c_str())] = @(v.c_str());
  opt.preprocessorMacros = defs;

  NSError *e = nil;
  id<MTLLibrary> lib = [dev_ newLibraryWithSource:@(src.c_str()) options:opt error:&e];
  if (!lib) {
    *err = "compiling " + file + " failed:\n" + std::string(e.localizedDescription.UTF8String);
    return nil;
  }
  id<MTLFunction> func = [lib newFunctionWithName:@(fn.c_str())];
  if (!func) {
    *err = "kernel " + fn + " not found in " + file;
    return nil;
  }
  id<MTLComputePipelineState> pso = [dev_ newComputePipelineStateWithFunction:func error:&e];
  if (!pso) {
    *err = "pipeline for " + fn + " failed:\n" + std::string(e.localizedDescription.UTF8String);
    return nil;
  }
  cache_[key] = pso;
  return pso;
}

id<MTLBuffer> Metal::alloc(size_t bytes) {
  size_t len = round_up(bytes ? bytes : 1, page_size());
  return [dev_ newBufferWithLength:len options:MTLResourceStorageModeShared];
}

id<MTLBuffer> Metal::wrap(void *ptr, size_t bytes) {
  if ((uintptr_t)ptr % page_size() || bytes % page_size()) return nil;
  return [dev_ newBufferWithBytesNoCopy:ptr length:bytes options:MTLResourceStorageModeShared deallocator:nil];
}

}  // namespace h3
