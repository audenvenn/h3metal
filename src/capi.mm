#include "capi.h"

#include "metal_rt.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <vector>

namespace {

thread_local std::string g_error;

void set_error(const std::string &msg) { g_error = msg; }

h3::Macros parse_macros(const char *s) {
  h3::Macros m;
  if (!s) return m;
  std::string all(s);
  size_t pos = 0;
  while (pos < all.size()) {
    size_t end = all.find(';', pos);
    if (end == std::string::npos) end = all.size();
    std::string item = all.substr(pos, end - pos);
    size_t eq = item.find('=');
    if (!item.empty()) {
      if (eq == std::string::npos) m[item] = "1";
      else m[item.substr(0, eq)] = item.substr(eq + 1);
    }
    pos = end + 1;
  }
  return m;
}

}  // namespace

struct h3_ctx {
  h3::Metal mt;
  id<MTLCommandBuffer> cb = nil;
  id<MTLComputeCommandEncoder> enc = nil;
  std::vector<id<MTLCommandBuffer>> pending;  // committed, not yet waited for
  bool profile = false;
  std::vector<double> prof;
  std::string name;

  explicit h3_ctx(const std::string &dir) : mt(dir) {}

  void begin() {
    if (!cb) {
      cb = [mt.queue() commandBuffer];
      enc = [cb computeCommandEncoder];
    }
  }
  void commit() {
    if (!cb) return;
    [enc endEncoding];
    [cb commit];
    pending.push_back(cb);
    cb = nil;
    enc = nil;
  }
};

struct h3_map {
  void *data = nullptr;
  size_t size = 0, mapped = 0;
  std::vector<id<MTLBuffer>> chunks;
};

extern "C" {

const char *h3_last_error(void) { return g_error.c_str(); }

h3_ctx *h3_open(const char *kernel_dir) {
  @autoreleasepool {
    h3_ctx *ctx = new h3_ctx(kernel_dir ? kernel_dir : "kernels");
    if (!ctx->mt.device() || !ctx->mt.queue()) {
      set_error("no Metal device");
      delete ctx;
      return nullptr;
    }
    ctx->name = ctx->mt.device().name.UTF8String;
    return ctx;
  }
}

void h3_close(h3_ctx *ctx) {
  if (!ctx) return;
  h3_wait(ctx);
  delete ctx;
}

const char *h3_device_name(h3_ctx *ctx) { return ctx->name.c_str(); }
uint64_t h3_recommended_working_set(h3_ctx *ctx) { return ctx->mt.device().recommendedMaxWorkingSetSize; }
uint64_t h3_max_buffer_length(h3_ctx *ctx) { return ctx->mt.device().maxBufferLength; }

void *h3_buffer(h3_ctx *ctx, uint64_t bytes) {
  id<MTLBuffer> b = ctx->mt.alloc(bytes);
  if (!b) {
    set_error("cannot allocate a buffer of " + std::to_string(bytes) + " bytes");
    return nullptr;
  }
  return (__bridge_retained void *)b;
}

void *h3_buffer_contents(void *buf) { return [(__bridge id<MTLBuffer>)buf contents]; }
uint64_t h3_buffer_length(void *buf) { return [(__bridge id<MTLBuffer>)buf length]; }
void h3_buffer_release(void *buf) {
  if (buf) CFRelease(buf);
}

h3_map *h3_map_open(h3_ctx *ctx, const char *path, uint64_t chunk_bytes) {
  const size_t page = h3::Metal::page_size();
  if (chunk_bytes == 0 || chunk_bytes % page) {
    set_error("chunk size must be a positive multiple of the page size");
    return nullptr;
  }
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    set_error(std::string("cannot open ") + path);
    return nullptr;
  }
  struct stat st;
  fstat(fd, &st);
  h3_map *m = new h3_map;
  m->size = (size_t)st.st_size;
  m->mapped = h3::Metal::round_up(m->size, page);
  m->data = mmap(nullptr, m->mapped, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (m->data == MAP_FAILED) {
    set_error(std::string("cannot map ") + path);
    delete m;
    return nullptr;
  }
  for (size_t off = 0; off < m->mapped; off += chunk_bytes) {
    size_t len = std::min<size_t>(chunk_bytes, m->mapped - off);
    id<MTLBuffer> b = ctx->mt.wrap((uint8_t *)m->data + off, len);
    if (!b) {
      set_error("cannot hand the mapping of " + std::string(path) + " to Metal");
      h3_map_close(m);
      return nullptr;
    }
    m->chunks.push_back(b);
  }
  return m;
}

uint64_t h3_map_size(h3_map *m) { return m->size; }
int h3_map_chunks(h3_map *m) { return (int)m->chunks.size(); }
void *h3_map_chunk(h3_map *m, int i) { return (__bridge void *)m->chunks[i]; }
const void *h3_map_data(h3_map *m) { return m->data; }

int h3_map_advise(h3_map *m, uint64_t offset, uint64_t length, int advice) {
  const size_t page = h3::Metal::page_size();
  uint64_t a = offset / page * page;
  uint64_t b = std::min<uint64_t>(h3::Metal::round_up(offset + length, page), m->mapped);
  int adv = advice == 1 ? MADV_WILLNEED : advice == 2 ? MADV_SEQUENTIAL : advice == 3 ? MADV_RANDOM : MADV_NORMAL;
  return madvise((uint8_t *)m->data + a, b - a, adv);
}

void h3_map_close(h3_map *m) {
  if (!m) return;
  m->chunks.clear();
  if (m->data && m->data != MAP_FAILED) munmap(m->data, m->mapped);
  delete m;
}

void *h3_pipeline(h3_ctx *ctx, const char *file, const char *fn, const char *macros) {
  @autoreleasepool {
    std::string err;
    id<MTLComputePipelineState> pso = ctx->mt.pipeline(file, fn, parse_macros(macros), &err);
    if (!pso) {
      set_error(err);
      return nullptr;
    }
    return (__bridge void *)pso;  // owned by the pipeline cache
  }
}

int h3_pipeline_max_threads(void *pso) {
  return (int)[(__bridge id<MTLComputePipelineState>)pso maxTotalThreadsPerThreadgroup];
}

int h3_dispatch(h3_ctx *ctx, void *pso, const uint32_t grid[3], const uint32_t tg[3], const h3_arg *bufs,
                int nbufs, const void *params, uint32_t params_len, int params_index, uint32_t tg_mem_bytes) {
  if (!pso) {
    set_error("dispatch without a pipeline");
    return -1;
  }
  ctx->begin();
  id<MTLComputeCommandEncoder> enc = ctx->enc;
  [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
  for (int i = 0; i < nbufs; i++) {
    if (bufs[i].buf) [enc setBuffer:(__bridge id<MTLBuffer>)bufs[i].buf offset:bufs[i].offset atIndex:i];
  }
  if (params && params_len) [enc setBytes:params length:params_len atIndex:params_index];
  if (tg_mem_bytes) [enc setThreadgroupMemoryLength:tg_mem_bytes atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2])
      threadsPerThreadgroup:MTLSizeMake(tg[0], tg[1], tg[2])];
  if (ctx->profile) {
    ctx->commit();
    id<MTLCommandBuffer> cb = ctx->pending.back();
    [cb waitUntilCompleted];
    ctx->prof.push_back(cb.GPUEndTime - cb.GPUStartTime);
  }
  return 0;
}

void h3_commit(h3_ctx *ctx) { ctx->commit(); }

double h3_wait(h3_ctx *ctx) {
  ctx->commit();
  double total = 0;
  bool failed = false;
  for (id<MTLCommandBuffer> cb : ctx->pending) {
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      failed = true;
      NSError *e = cb.error;
      set_error(std::string("GPU command buffer failed: ") + (e ? e.localizedDescription.UTF8String : "unknown"));
    }
    total += cb.GPUEndTime - cb.GPUStartTime;
  }
  ctx->pending.clear();
  return failed ? -1.0 : total;
}

void h3_set_profile(h3_ctx *ctx, int on) {
  h3_wait(ctx);
  ctx->profile = on != 0;
  ctx->prof.clear();
}

int h3_profile_take(h3_ctx *ctx, double *seconds, int max) {
  int n = std::min<int>(max, (int)ctx->prof.size());
  for (int i = 0; i < n; i++) seconds[i] = ctx->prof[i];
  ctx->prof.clear();
  return n;
}

}  // extern "C"
