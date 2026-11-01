// Checks the CUDA plugin's CPU fallback with views of one device tensor. A
// region gets two rank-1 f32 views, doubles the first one, and does not touch
// the second one. The plugin copies the views to the host for the call and
// back after it.
//
// Usage: gpu_view_copy_check            runs the cases below
//        gpu_view_copy_check <offset>   gives the second view that offset;
//                                       the plugin must stop the program

#include "vx_hardware_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

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

// The region. The plugin finds it by the name at the start of the payload.
extern "C" void _vx_ciface_vx_check_views(Desc1 *a, Desc1 *b) {
  float *data = (float *)a->aligned + a->offset;
  for (int64_t j = 0; j < a->sizes[0]; ++j) {
    data[j * a->strides[0]] *= 2;
  }
  (void)b;
}

// No `image=` and no `kind=matmul`, so the plugin runs the region on the CPU.
static const char kPayload[] = "vx_check_views\0abi=1\0";

static int failures = 0;

static int dispatch(Desc1 *a, Desc1 *b) {
  void *p_a = a, *p_b = b;
  void *device_args[2] = {&p_a, &p_b};
  const int32_t tags[2] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1)};
  return (int)vx_plugin_dispatch_async(kPayload, sizeof(kPayload), device_args,
                                       tags, 2);
}

// Put `before` in device memory, dispatch the two views, and compare the
// tensor with `expected`.
static void run_case(const char *name, const std::vector<float> &before,
                     const std::vector<float> &expected, Desc1 a, Desc1 b) {
  size_t bytes = before.size() * sizeof(float);
  float *device = nullptr;
  if (cudaMalloc(&device, bytes) != cudaSuccess ||
      cudaMemcpy(device, before.data(), bytes, cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    fprintf(stderr, "FAIL: %s: could not put the tensor in device memory\n",
            name);
    ++failures;
    return;
  }
  a.allocated = a.aligned = device;
  b.allocated = b.aligned = device;
  int64_t a_offset = a.offset, b_offset = b.offset;
  if (dispatch(&a, &b) != 1) {
    fprintf(stderr, "FAIL: %s: the dispatch did not succeed\n", name);
    ++failures;
  }
  if (a.aligned != device || b.aligned != device || a.allocated != device ||
      b.allocated != device || a.offset != a_offset || b.offset != b_offset) {
    fprintf(stderr, "FAIL: %s: a descriptor was not restored\n", name);
    ++failures;
  }
  std::vector<float> after(before.size());
  cudaMemcpy(after.data(), device, bytes, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < after.size(); ++i) {
    if (after[i] != expected[i]) {
      fprintf(stderr, "FAIL: %s: element %zu is %g, expected %g\n", name, i,
              (double)after[i], (double)expected[i]);
      ++failures;
    }
  }
  cudaFree(device);
}

int main(int argc, char **argv) {
  if (argc == 2) {
    float *device = nullptr;
    cudaMalloc(&device, 8 * sizeof(float));
    Desc1 a = {device, device, 0, {4}, {1}};
    Desc1 b = {device, device, atoll(argv[1]), {4}, {1}};
    dispatch(&a, &b);
    fprintf(stderr, "FAIL: the dispatch with offset %s returned\n", argv[1]);
    return 1;
  }

  // Row 0 and row 1 of a 2x4 tensor.
  run_case("two rows", {1, 2, 3, 4, 5, 6, 7, 8}, {2, 4, 6, 8, 5, 6, 7, 8},
           {nullptr, nullptr, 0, {4}, {1}}, {nullptr, nullptr, 4, {4}, {1}});
  // The even and the odd columns of a 2x4 tensor.
  run_case("even and odd columns", {1, 2, 3, 4, 5, 6, 7, 8},
           {2, 2, 6, 4, 10, 6, 14, 8}, {nullptr, nullptr, 0, {4}, {2}},
           {nullptr, nullptr, 1, {4}, {2}});
  // The same columns, with the even columns second. The region doubles the
  // odd columns.
  run_case("odd and even columns", {1, 2, 3, 4, 5, 6, 7, 8},
           {1, 4, 3, 8, 5, 12, 7, 16}, {nullptr, nullptr, 1, {4}, {2}},
           {nullptr, nullptr, 0, {4}, {2}});
  // One tensor passed twice.
  run_case("one tensor twice", {1, 2, 3, 4}, {2, 4, 6, 8},
           {nullptr, nullptr, 0, {4}, {1}}, {nullptr, nullptr, 0, {4}, {1}});
  // The last row of a 1024x4 tensor, and the row before it.
  std::vector<float> big(1024 * 4, 1.0f), big_after = big;
  for (int j = 0; j < 4; ++j) {
    big_after[1023 * 4 + j] = 2;
  }
  run_case("last row", big, big_after, {nullptr, nullptr, 1023 * 4, {4}, {1}},
           {nullptr, nullptr, 1022 * 4, {4}, {1}});

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
