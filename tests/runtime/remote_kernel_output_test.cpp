//===- remote_kernel_output_test.cpp - A worker's writes reach the host -*- C++
//-*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host calls vx_remote_dispatch on buffers in its own memory. The client
// copies each buffer to the worker, the worker runs the kernel on the copies,
// and the client must copy the written values back into the host buffers.
//
// Three dispatches, over one socket to a fake worker in a child process:
//
//   * A kernel the compiler generated (the payload carries `image=`) writes two
//     host buffers. Both must hold the kernel's values afterwards.
//   * The same kernel, with one output already on the worker. That output must
//     stay on the worker, and the other output must still reach the host.
//   * A matmul with `outkind=buffer`, which fills one host buffer.
//
// The fake worker does not load the image. It runs a fixed function in its
// place: out1[i] = 2 * in[i] and out2[i] = in[i] + 100.
//
// Driven by tests/integration_test/remote_kernel_output_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_remote_client.h"
#include "test_worker.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using namespace vx_test;

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

/* --- the child: a fake worker -------------------------------------------- */

uint8_t child_body[1 << 20];
uint8_t child_reply[4096];

float *resolve_f32(uint64_t handle) {
  vx_remote_ref ref;
  if (!vx_remote_resolve(&worker_table, handle, &ref)) {
    return nullptr;
  }
  return (float *)((char *)ref.region->remote + ref.offset);
}

// The stand-in for a generated kernel. Arguments are in, out1, out2, each
// four f32 values.
int run_fake_kernel(const vx_wire_arg *args, int64_t num_args) {
  if (num_args != 3) {
    return 0;
  }
  const float *in = resolve_f32(args[0].handle);
  float *out1 = resolve_f32(args[1].handle);
  float *out2 = resolve_f32(args[2].handle);
  if (!in || !out1 || !out2) {
    return 0;
  }
  for (int i = 0; i < 4; ++i) {
    out1[i] = 2 * in[i];
    out2[i] = in[i] + 100;
  }
  return 1;
}

void run_worker(int fd) {
  worker_reset();

  for (;;) {
    uint32_t type = 0;
    uint64_t len = 0;
    int rc = vx_transport_recv(fd, &type, child_body, sizeof(child_body), &len);
    if (rc == VX_TRANSPORT_EOF) {
      _exit(0);
    }
    if (rc != VX_TRANSPORT_OK) {
      _exit(3);
    }

    vx_wire_reader r;
    vx_wire_writer w;
    vx_wire_reader_init(&r, child_body, (size_t)len);
    vx_wire_writer_init(&w, child_reply, sizeof(child_reply));

    if (type == VX_WIRE_TRANSFER) {
      vx_wire_transfer t;
      uint64_t handle = vx_wire_get_transfer(&r, &t) ? worker_transfer(&t) : 0;
      vx_wire_put_u64(&w, handle);
      if (vx_transport_send(fd, VX_WIRE_TRANSFER, child_reply, w.len) !=
          VX_TRANSPORT_OK) {
        _exit(4);
      }
    } else if (type == VX_WIRE_DISPATCH) {
      vx_wire_dispatch d;
      vx_wire_arg args[8];
      vx_wire_result results[4];
      int64_t num_results = 0;
      int ok = vx_wire_get_dispatch(&r, &d) && d.num_args <= 8;
      for (int64_t i = 0; ok && i < d.num_args; ++i) {
        ok = vx_wire_get_arg(&r, &args[i]);
      }
      if (ok) {
        if (vx_payload_field(d.payload, (size_t)d.payload_len, "image=")) {
          ok = run_fake_kernel(args, d.num_args);
        } else {
          ok = worker_dispatch(&d, args, results, &num_results);
        }
      }
      vx_wire_put_results(&w, ok ? 0 : -1, results, ok ? num_results : 0);
      if (vx_transport_send(fd, VX_WIRE_DISPATCH, child_reply, w.len) !=
          VX_TRANSPORT_OK) {
        _exit(5);
      }
    } else if (type == VX_WIRE_FETCH) {
      uint64_t handle = 0, nbytes = 0;
      vx_remote_ref ref;
      if (!vx_wire_get_fetch(&r, &handle, &nbytes) ||
          !vx_remote_resolve(&worker_table, handle, &ref) ||
          nbytes > ref.region->size - ref.offset) {
        rc = vx_transport_send(fd, VX_WIRE_FETCH, nullptr, 0);
      } else {
        rc = vx_transport_send(fd, VX_WIRE_FETCH,
                               (const char *)ref.region->remote + ref.offset,
                               nbytes);
      }
      if (rc != VX_TRANSPORT_OK) {
        _exit(6);
      }
    } else if (type == VX_WIRE_FREE) {
      // The test arena never reuses memory, so a FREE only needs its reply.
      if (vx_transport_send(fd, VX_WIRE_FREE, nullptr, 0) != VX_TRANSPORT_OK) {
        _exit(7);
      }
    } else {
      _exit(8);
    }
  }
}

/* --- the parent: the host ------------------------------------------------ */

// A memref descriptor as generated code lays it out, for rank 1 and rank 2.
struct Desc1 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};
struct Desc2 {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

Desc1 vector_of_four(void *data) {
  Desc1 d = {data, data, 0, {4}, {1}};
  return d;
}

Desc2 two_by_two(void *data) {
  Desc2 d = {data, data, 0, {2, 2}, {2, 1}};
  return d;
}

uint8_t host_scratch[1 << 20];

// A payload in the shape the compiler writes for a kernel it generated. The
// image bytes are not real PTX, because the fake worker never loads them.
const char kKernelPayload[] = "vx_gpu_kernel_0\0abi=1\0topo=500\0"
                              "format=ptx\0image=not-real-ptx\0";

void test_a_generated_kernel_writes_both_host_outputs(int fd) {
  float in[4] = {1, 2, 3, 4};
  float out1[4] = {0, 0, 0, 0};
  float out2[4] = {0, 0, 0, 0};
  Desc1 d_in = vector_of_four(in);
  Desc1 d_out1 = vector_of_four(out1);
  Desc1 d_out2 = vector_of_four(out2);
  void *p_in = &d_in, *p_out1 = &d_out1, *p_out2 = &d_out2;
  void *device_args[3] = {&p_in, &p_out1, &p_out2};
  const int32_t tags[3] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1)};
  vx_wire_arg args[3];

  check(vx_remote_dispatch(fd, kKernelPayload, sizeof(kKernelPayload),
                           device_args, tags, 3, args, host_scratch,
                           sizeof(host_scratch)) == 1,
        "the generated kernel's dispatch succeeds");
  for (int i = 0; i < 4; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "first output [%d] == %g", i,
             (double)(2 * in[i]));
    check(out1[i] == 2 * in[i], what);
    snprintf(what, sizeof(what), "second output [%d] == %g", i,
             (double)(in[i] + 100));
    check(out2[i] == in[i] + 100, what);
  }
  check(in[0] == 1 && in[1] == 2 && in[2] == 3 && in[3] == 4,
        "the input is unchanged");
}

void test_an_output_on_the_worker_stays_there(int fd) {
  float in[4] = {1, 2, 3, 4};
  float zeros[4] = {0, 0, 0, 0};
  float out2[4] = {0, 0, 0, 0};

  // Put the first output on the worker before the dispatch, as `transfer`
  // does in a program. Its descriptor then holds a handle, not a pointer.
  uint64_t resident = vx_remote_transfer(fd, zeros, sizeof(zeros), VX_DTYPE_F32,
                                         host_scratch, sizeof(host_scratch));
  check(resident != 0, "the resident output was placed on the worker");

  Desc1 d_in = vector_of_four(in);
  Desc1 d_out1 = vector_of_four((void *)(uintptr_t)resident);
  Desc1 d_out2 = vector_of_four(out2);
  void *p_in = &d_in, *p_out1 = &d_out1, *p_out2 = &d_out2;
  void *device_args[3] = {&p_in, &p_out1, &p_out2};
  const int32_t tags[3] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 1)};
  vx_wire_arg args[3];

  check(vx_remote_dispatch(fd, kKernelPayload, sizeof(kKernelPayload),
                           device_args, tags, 3, args, host_scratch,
                           sizeof(host_scratch)) == 1,
        "the dispatch with a resident output succeeds");
  check(d_out1.aligned == (void *)(uintptr_t)resident,
        "the resident output's descriptor still holds its handle");

  float fetched[4] = {0, 0, 0, 0};
  check(vx_remote_fetch(fd, resident, fetched, sizeof(fetched)),
        "the resident output can be fetched");
  for (int i = 0; i < 4; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "resident output [%d] == %g", i,
             (double)(2 * in[i]));
    check(fetched[i] == 2 * in[i], what);
    snprintf(what, sizeof(what), "host output [%d] == %g", i,
             (double)(in[i] + 100));
    check(out2[i] == in[i] + 100, what);
  }
  check(vx_remote_free(fd, resident), "the resident output is freed");
}

void test_a_matmul_fills_a_host_buffer(int fd) {
  // A = [1 2; 3 4], B = [5 6; 7 8], A * B = [19 22; 43 50].
  float a[4] = {1, 2, 3, 4};
  float b[4] = {5, 6, 7, 8};
  float c[4] = {0, 0, 0, 0};
  const float expected[4] = {19, 22, 43, 50};
  Desc2 d_a = two_by_two(a);
  Desc2 d_b = two_by_two(b);
  Desc2 d_c = two_by_two(c);
  void *p_a = &d_a, *p_b = &d_b, *p_c = &d_c;
  void *device_args[3] = {&p_a, &p_b, &p_c};
  const int32_t tags[3] = {VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2),
                           VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2)};
  static const char payload[] = "vx_npu_kernel_1\0kind=matmul\0"
                                "roles=a:0,b:1,out:2\0outkind=buffer\0"
                                "topo=1113\0toponame=DecodeWorker\0";
  vx_wire_arg args[3];

  check(vx_remote_dispatch(fd, payload, sizeof(payload), device_args, tags, 3,
                           args, host_scratch, sizeof(host_scratch)) == 1,
        "the matmul's dispatch succeeds");
  for (int i = 0; i < 4; ++i) {
    char what[64];
    snprintf(what, sizeof(what), "C[%d] == %g", i, (double)expected[i]);
    check(c[i] == expected[i], what);
  }
}

} // namespace

int main() {
  int sv[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
    fprintf(stderr, "FAIL: socketpair\n");
    return 1;
  }
  pid_t pid = fork();
  if (pid < 0) {
    fprintf(stderr, "FAIL: fork\n");
    return 1;
  }
  if (pid == 0) {
    close(sv[0]);
    run_worker(sv[1]);
  }
  close(sv[1]);

  test_a_generated_kernel_writes_both_host_outputs(sv[0]);
  test_an_output_on_the_worker_stays_there(sv[0]);
  test_a_matmul_fills_a_host_buffer(sv[0]);

  close(sv[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "the fake worker exited cleanly");

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
