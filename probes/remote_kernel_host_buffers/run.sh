#!/usr/bin/env bash
# Compiles each probe program to LLVM IR and reports what the compiler did.
# Run it from the repository root after `cargo build --release --bin vxc`.
#
# For each case it prints:
#   - the exit code of vxc and its error lines,
#   - how many launches carry a GPU kernel (`image=`),
#   - how many calls to vx_plugin_alloc_and_transfer the IR has.
# A case reaches the client path in question when it compiles, has `image=`,
# and its kernel arguments are host tensors (no transfer before the launch).

set -u

VXC=target/release/vxc
DIR=probes/remote_kernel_host_buffers
OUT=$DIR/out
mkdir -p "$OUT"

# name | program | vxc options
CASES=(
  "01_fleet_a100|01_host_tensors_with_no_transfer.vx|--machine fleet/a100-80.vx"
  "01_fleet_a100_host_default|01_host_tensors_with_no_transfer.vx|--host default --machine fleet/a100-80.vx"
  "01_no_machine|01_host_tensors_with_no_transfer.vx|"
  "02_shared_memory|02_host_tensors_on_a_machine_with_shared_memory.vx|--machine $DIR/shared_host_memory_machine.vx"
  "02_shared_memory_host_default|02_host_tensors_on_a_machine_with_shared_memory.vx|--host default --machine $DIR/shared_host_memory_machine.vx"
  "03_fleet_a100|03_host_tensors_passed_as_hbm_parameters.vx|--host default --machine fleet/a100-80.vx"
  "04_fleet_a100|04_kernel_after_a_transfer.vx|--host default --machine fleet/a100-80.vx"
)

for c in "${CASES[@]}"; do
  IFS='|' read -r name program options <<<"$c"
  log=$OUT/$name.log
  ll=$OUT/$name.ll
  # shellcheck disable=SC2086
  "$VXC" "$DIR/$program" $options --emit-llvm -o "$ll" >"$log" 2>&1
  code=$?
  # The IR may go to the -o file or to the log, so count in both.
  touch "$ll"
  images=$(cat "$ll" "$log" | grep -o 'image=' | wc -l)
  transfers=$(cat "$ll" "$log" | grep -c 'call .*@vx_plugin_alloc_and_transfer')
  dispatches=$(cat "$ll" "$log" | grep -c 'call .*@vx_plugin_dispatch_async')
  echo "=== $name: vxc $program $options"
  echo "exit code: $code"
  echo "launches with image=: $images"
  echo "calls to vx_plugin_dispatch_async: $dispatches"
  echo "calls to vx_plugin_alloc_and_transfer: $transfers"
  echo "--- compiler messages"
  grep -iE 'error|warning' "$log" | head -20
  echo
done
