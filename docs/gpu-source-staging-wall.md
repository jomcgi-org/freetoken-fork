# GPU source staging wall comparison

Use `bench/pi-decode-prefix-wall-driver.py --gpu-source-staging
--fixed-continuation --run-id astra-pi-agentic-<unique-name>
--output-dir /private/tmp/<unique-name>` from a clean linked worktree.
Add `--preflight` to check the configuration without pausing serving.

Both arms use the same pinned GPU-source runtime revision and all four native
extensions. Only `--moe-gpu-source pinned|staged` changes. Decode prefix snapshots,
token traces and invasive MoE diagnostics stay off. The controller runs pinned,
staged, staged, pinned, each with a fresh server, one warmup and two measured
three-turn conversations. The host page cache is retained.

The startup gate preserves GPU compute selection, HOT residency, CPU executor
configuration, expert slots, FP8 KV capacity and graph batch size. It separately
checks the intended transition from 20 pinned and 28 file-backed layers to
48 file-backed layers, with precisely the original 20 GPU layers staged. A
different automatically selected GPU cache size aborts the run.

Worker RSS categories, locked memory, swap and OS cache/available memory are
sampled before and after the client. These snapshots include the warmup and are
outside measured requests. RSS snapshots do not measure peak resident memory;
`VmHWM` is the process lifetime high-water mark. System counters include other
processes, and file RSS does not include all unmapped file cache. The worker I/O
delta also includes warmup. Neither file-backed capacity nor reduced pinned
capacity alone establishes net RAM savings.

Summarize with `bench/pi-agentic-runtime-summary.py <results-dir>
--gpu-source-staging`. The summary requires matching request bodies, answer
bytes and token counts, including warmups, before reporting a wall reduction.
It retains both execution orders, failures and separate first-request and
continuation times. Matching this synthetic workload does not establish broad
model quality equivalence.

The existing remote lease and systemd restoration hook restore original
serving on cancellation or lost client heartbeats. Run the recovery probe and
check exclusive GPU ownership before timing; verify original serving and
review service errors after all arms finish. Keep measured artifacts private.
