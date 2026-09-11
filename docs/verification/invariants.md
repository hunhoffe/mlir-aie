<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Invariant registry

What the toolchain promises for designs in the safe subset, and where each
promise is checked. One row per invariant; every checker that lands adds or
updates rows here, with the mutation set its tests were shown to catch. The
levels are the layers a design passes through: IRON (Python), ObjectFIFO
(pre-lowering MLIR), lowered `aie` IR, the runtime sequence, and the host.

IDs are stable: `K-` rows are kernel-level facts a `KernelContract`
declares, checked by the harness on hardware; MLIR-level checkers will use
the diagnostic scheme proposed with the first one.

| ID | Statement | Level | Checker | Tests | Mutation set |
|----|-----------|-------|---------|-------|--------------|
| K-STATE-1 | A kernel's output does not depend on the rounding or saturation register the core held when it was called, unless its contract names the mode a design must set first (`rounding_mode` / `saturation_mode`). `unspecified` claims independence; `sets_own` claims the source writes the register on every path. | IRON / hardware | `aie.utils.kernel_harness.judge_dirty` on a design built with `core_state` (the preset runs before the contract's own setters) | `test/python/npu/test_kernels_e2e.py::test_kernel_core_state` (device, nightly `kernelCoreState.yml`); `test/python/test_kernel_core_state.py` (host: judge and design order) | M1-M6 below |
| K-STATE-2 | A kernel runs in the state its contract names: the design's setters take effect after any preset, and the probe reads them back before the first call. | IRON / hardware | `kernel_harness.judge_state` on the probe's `before` reading | as K-STATE-1 | M7-M9 |
| K-STATE-3 | A kernel leaves each register as its contract says (`leaves_rounding` / `leaves_saturation`): `preserves`, or the mode it sets and does not restore. The next kernel on the core inherits it. | IRON / hardware | `kernel_harness.judge_state` on the probe's `after` reading; `test_register_effects_follow_the_sources` ties the claims to the C++ | as K-STATE-1 | M10-M14 |
| K-STATE-4 | A fresh core holds `BOOT_ROUNDING` (`floor`) and `BOOT_SATURATION` (`none`); every contract that names nothing assumes this. | hardware | `test_kernel_core_state::test_core_boot_state` (a probe-only design that presets nothing) | device, nightly | M15 |

## Mutation set for K-STATE-1 to K-STATE-4

Each line is a plausible one-line regression and the test that catches it.

| # | Regression | Caught by |
|---|------------|-----------|
| M1 | `judge_dirty` compares under the contract tolerance instead of bit-for-bit | `test_judge_dirty_fails_any_difference_whatever_the_tolerance` (one ulp, `-0.0`, a NaN payload) |
| M2 | `judge_dirty` compares values, so equal NaNs count as different or different NaNs as equal | `test_judge_dirty_passes_bit_identical_output_including_nans`, the NaN-payload case of M1 |
| M3 | The preset setters run *after* the contract's own setters (the preset overrides the kernel's declared mode) | `test_design_presets_then_sets_then_probes_then_runs` (call order) |
| M4 | The preset is dropped when the contract names a mode | `test_design_presets_then_sets_then_probes_then_runs` (`set_rounding_ceil` still first) |
| M5 | Only the rounding register is preset; saturation is ignored | `test_design_presets_then_sets_then_probes_then_runs`, `test_judge_dirty_names_the_fix_per_claim` |
| M6 | The sweep runs the smoke cases, missing `mm` variants, or runs one build twice | `test_distinct_kernels_cover_every_build_once_with_the_smallest_case` |
| M6b | The trimmed preset list loses a register (no saturation preset, or no rounding preset other than the boot state) | `test_trimmed_presets_touch_both_registers` |
| M7 | The probe's `before` reading is not judged (a setter that does nothing passes) | `test_judge_state_reports_a_preset_or_setter_that_did_not_take` |
| M8 | The probe runs before the setters | `test_design_presets_then_sets_then_probes_then_runs` (probe is call 4) |
| M9 | The state tile is not poisoned, so a probe that never runs reads as the boot state | `test_decode_core_state` (poison decodes to `None`, marker missing), `test_judge_state_reports_a_probe_that_never_ran` |
| M10 | `expected_state` uses the preset for `after` when the contract leaves a mode | `test_expected_state` (`SETS_AND_LEAVES` row) |
| M11 | A contract can claim `leaves_rounding="conv_even"` without `sets_own` | `test_contract_validates_saturation_and_leaves` |
| M12 | A conv factory drops its `leaves_saturation="saturate"` while the source still calls `set_saturation` | `test_register_effects_follow_the_sources` |
| M13 | A factory claims `sets_own` saturation for a source that never writes it | `test_register_effects_follow_the_sources` |
| M14 | `read_core_state.cc` renumbers a mode, or the Python table reorders one | `test_probe_codes_match_the_python_tables` |
| M15 | `BOOT_ROUNDING` is changed to a mode the hardware does not boot in | `test_core_boot_state` (device) |
| M16 | The second (after) probe reading is dropped from the design | `test_design_presets_then_sets_then_probes_then_runs` (`read_core_state` twice, last call) |
| M17 | `design(core_state=...)` reuses the un-probed generators, so the state tensor is never drained | `test_matrix_and_packed_designs_probe_too` (`memref<8xi32>` in every family) |

Not caught on a host, and why: a kernel that reads the register on only
some data paths passes the host tests and fails only on the device sweep,
which is what the sweep is for; and the AIE API enumerator names in the two
generic `.cc` files are checked only by compiling them (the static
kernel-check workflow does, nightly, for aie2 and aie2p).
