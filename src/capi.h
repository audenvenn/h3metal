// h3metal C API: a small GPU kernel launcher.
//
// The model itself is written in Python (h3metal/); this library owns what
// Python cannot do cheaply: the Metal device, run-time compiled kernels,
// buffers, read-only file mappings handed to the GPU without copying, and the
// encoding of dispatches into command buffers. Dispatches are recorded into
// one open command buffer until h3_commit(); h3_wait() blocks until all
// committed work is done.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct h3_ctx h3_ctx;
typedef struct h3_map h3_map;

// A buffer handle (an opaque, retained MTLBuffer) and a byte offset into it.
typedef struct {
  void *buf;
  uint64_t offset;
} h3_arg;

// Message of the last failed call on this thread (empty if none).
const char *h3_last_error(void);

h3_ctx *h3_open(const char *kernel_dir);
void h3_close(h3_ctx *ctx);
const char *h3_device_name(h3_ctx *ctx);
uint64_t h3_recommended_working_set(h3_ctx *ctx);
uint64_t h3_max_buffer_length(h3_ctx *ctx);

// Shared-storage buffers, zero-filled. The CPU sees them at h3_buffer_contents().
void *h3_buffer(h3_ctx *ctx, uint64_t bytes);
void *h3_buffer_contents(void *buf);
uint64_t h3_buffer_length(void *buf);
void h3_buffer_release(void *buf);

// A read-only mapping of a file, wrapped as Metal buffers of at most
// `chunk_bytes` each (a multiple of the page size). A tensor must not cross a
// chunk boundary; the file writer guarantees that.
h3_map *h3_map_open(h3_ctx *ctx, const char *path, uint64_t chunk_bytes);
uint64_t h3_map_size(h3_map *m);
int h3_map_chunks(h3_map *m);
void *h3_map_chunk(h3_map *m, int i);
const void *h3_map_data(h3_map *m);
// advice: 0 normal, 1 will need (read ahead), 2 sequential, 3 random. Returns 0 on success.
int h3_map_advise(h3_map *m, uint64_t offset, uint64_t length, int advice);
void h3_map_close(h3_map *m);

// Compiles (or fetches from the cache) kernel `fn` of kernels/<file>, with
// preprocessor macros given as "NAME=value;NAME2=value2". NULL on failure.
void *h3_pipeline(h3_ctx *ctx, const char *file, const char *fn, const char *macros);
int h3_pipeline_max_threads(void *pso);

// Records one dispatch of `grid` threadgroups of `tg` threads. Buffers are bound
// at indices 0..nbufs-1 (a NULL buf leaves that index unbound); `params`, if
// any, are copied inline at index `params_index`. Returns 0 on success.
int h3_dispatch(h3_ctx *ctx, void *pso, const uint32_t grid[3], const uint32_t tg[3], const h3_arg *bufs,
                int nbufs, const void *params, uint32_t params_len, int params_index, uint32_t tg_mem_bytes);

// Ends the open command buffer and submits it, without waiting.
void h3_commit(h3_ctx *ctx);
// Commits, then waits for all submitted work. Returns the GPU time in seconds
// of the command buffers completed since the previous wait, or a negative
// value if one of them failed (see h3_last_error).
double h3_wait(h3_ctx *ctx);
// With profiling on, every dispatch is submitted in its own command buffer and
// timed; h3_profile_take() returns the times recorded since the last call.
void h3_set_profile(h3_ctx *ctx, int on);
int h3_profile_take(h3_ctx *ctx, double *seconds, int max);

#ifdef __cplusplus
}
#endif
