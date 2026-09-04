# Issue 18 CUDA verification

Run these commands on a CUDA machine with the CPU MoE extension built. They use
the module order that exposed the leaked current CUDA stream.

## Reproduce before the fix

Run this against `org/feat/moe-disk-tier`, or against this branch before applying
the test isolation patch:

```bash
pytest -vv \
  tests/moe/test_step_timing.py \
  tests/moe/test_cpu_moe.py \
  tests/moe/test_disk_tier.py::test_disk_hot_cold_split_matches_pure_cpu_decode
```

The final test should fail with `Tensor-likes are not close!`. As a control, it
should pass alone:

```bash
pytest -vv \
  tests/moe/test_disk_tier.py::test_disk_hot_cold_split_matches_pure_cpu_decode
```

## Confirm after the fix

Run the same triggering order against `fix/parity-order`:

```bash
pytest -vv \
  tests/moe/test_step_timing.py \
  tests/moe/test_cpu_moe.py \
  tests/moe/test_disk_tier.py::test_disk_hot_cold_split_matches_pure_cpu_decode
```

Then run the reported full tier selection:

```bash
pytest -vv \
  tests/moe/test_step_timing.py \
  tests/engine/test_moe_cpu_layers.py \
  tests/scheduler/test_scheduler_status.py \
  tests/server/test_parser_auto_selection.py \
  tests/moe/test_cpu_moe.py \
  tests/moe/test_cpu_moe_dedup.py \
  tests/moe/test_hot_adapt.py \
  tests/moe/test_hot_plan_persistence.py \
  tests/moe/test_disk_tier.py \
  tests/moe/test_moe_collect_stats.py \
  tests/scheduler/test_kv_ladder.py
```

Expected result: the focused sequence and the full tier selection pass, with no
failure in `test_disk_hot_cold_split_matches_pure_cpu_decode`.
