// Checks the CUDA plugin's CPU fallback with two row views of one device
// tensor. A region gets A = row 0 and B = row 1 of a 2x4 f32 tensor in device
// memory, and doubles row 0 through A. The plugin copies each view to the host
// for the call and back after it. If B's copy also holds row 0, it goes back
// after A's and puts the old row 0 back.

#include "vx_hardware_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" uint64_t vx_plugin_dispatch_async(const void *binary_payload,
                                             size_t payload_size,
                                             void **device_args,
                                             const int32_t *arg_tags,
                                             int64_t num_args);

struct Desc1 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

// The region. The plugin finds it by the name at the start of the payload. Its
// arguments are the two views, and it writes only A.
extern "C" void _vx_ciface_vx_check_row_views(Desc1 *a, Desc1 *b) {
  float *data = (float *)a->aligned;
  for (int64_t j = 0; j < a->sizes[0]; ++j) {
    data[a->offset + j * a->strides[0]] *= 2;
  }
  (void)b;
}

// No `image=` and no `kind=matmul`, so the plugin runs the region on the CPU.
static const char kPayload[] = "vx_check_row_views\0abi=1\0";

int main(int argc, char **argv) {
  const float before[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  if (argc == 2) {
    // Give one view the offset in argv[1]. A broken offset must stop the
    // program before malloc.
    float *device = nullptr;
    cudaMalloc(&device, sizeof(before));
    Desc1 a = {device, device, 0, {4}, {1}};
    Desc1 b = {device, device, atoll(argv[1]), {4}, {1}};
    void *p_a = &a, *p_b = &b;
    void *device_args[2] = {&p_a, &p_b};
    const int32_t tags[2] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                             VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1)};
    vx_plugin_dispatch_async(kPayload, sizeof(kPayload), device_args, tags, 2);
    fprintf(stderr, "FAIL: the dispatch with offset %s returned\n", argv[1]);
    return 1;
  }
  const float expected[8] = {2, 4, 6, 8, 5, 6, 7, 8};
  float *device = nullptr;
  if (cudaMalloc(&device, sizeof(before)) != cudaSuccess ||
      cudaMemcpy(device, before, sizeof(before), cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    fprintf(stderr, "FAIL: could not put the tensor in device memory\n");
    return 2;
  }

  Desc1 a = {device, device, 0, {4}, {1}};
  Desc1 b = {device, device, 4, {4}, {1}};
  void *p_a = &a, *p_b = &b;
  void *device_args[2] = {&p_a, &p_b};
  const int32_t tags[2] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1)};
  if (vx_plugin_dispatch_async(kPayload, sizeof(kPayload), device_args, tags,
                               2) != 1) {
    fprintf(stderr, "FAIL: the dispatch did not succeed\n");
    return 1;
  }

  int failures = 0;
  if (a.aligned != device || b.aligned != device) {
    fprintf(stderr, "FAIL: a view does not point at device memory again\n");
    ++failures;
  }
  float after[8];
  if (cudaMemcpy(after, device, sizeof(after), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    fprintf(stderr, "FAIL: could not read the tensor back\n");
    return 2;
  }
  for (int i = 0; i < 8; ++i) {
    if (after[i] != expected[i]) {
      fprintf(stderr, "FAIL: element %d is %g, expected %g\n", i,
              (double)after[i], (double)expected[i]);
      ++failures;
    }
  }
  cudaFree(device);
  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
