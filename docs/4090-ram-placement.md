# RTX 4090 RAM placement experiment

On node-4's 61.91 GiB host, the automatic memory governor pins 20 expert
layers (26.44 GiB), leaving 28 DISK layers (37.02 GiB). Setting
`FREETOKEN_PIN_BUDGET_GB=40` pins 30 layers (39.66 GiB) and leaves 18 DISK
layers (23.80 GiB). The same 6 GiB protected GPU budget then covers 129
experts per DISK layer instead of 82. Router selections are unchanged.

The tradeoff is less page cache and more GPU execution. The 40 GiB setting
fits the governor's ceiling with its existing 9.29 GiB reserve; the derived
pager budget was 7.77 to 8.22 GiB across its two starts. No memory-governor
defaults were changed.

## Non-debug results, 2026-09-05

Four starts ran in order: 40 GiB, automatic, automatic, 40 GiB. Each server
ran the same eight fidelity checks, six timing warmups, six prefill pairs,
and two 192-token decode pairs. Both CPU/GPU prefill schedules ran in each
server. Selective transfers up to 128 tokens were enabled in every arm.
HOT adaptation, HOT plan persistence, disk KV reuse, MoE diagnostic collection,
and GPU timing were off; the KV cache was naive. CPU threads remained 14,
PLE used io_uring, expert slots remained 4,045, and KV capacity was 65,536.

Using only the serial schedule in each budget, mean client time to first
token was:

| Prompt tokens | Automatic (s) | 40 GiB (s) | Reduction |
| ---: | ---: | ---: | ---: |
| 76 | 1.844 | 1.560 | 15.4% |
| 524 | 5.781 | 4.424 | 23.5% |
| 2,060 | 18.785 | 12.053 | 35.8% |

Each cell contains four requests across two server starts. Both starts show
the same direction at every size. The concurrent schedule also showed a RAM
benefit, with reductions of 17.0%, 24.3%, and 34.8%. These are results for
this static placement and workload, not a universal optimum for RAM usage.

Decode request wall time also fell in this small sample, but the generated
answers differ across budgets and repeated answers warm the expert cache.
Retain the raw timings; do not use this sample to declare a general decode
improvement or unchanged model quality.

## Quality evidence and limits

Every start scored 7/8 on the same exact-answer checks. The CPU-code question
returned `108` instead of the correct `68` at both budgets. The initial 40
GiB attempt stopped at this failure before timing anything; an automatic
baseline then reproduced all eight answers, including that error. Subsequent
starts required no new failure relative to that baseline, and all eight
answers continued to match exactly. The failed case remains in the record.

All 32 timing pairs produced identical text and usage between serial and
concurrent scheduling. Across RAM budgets, the longer answers differ:
placement changes which routes use CPU W4A8 versus GPU arithmetic. The eight
checks are too small to establish unchanged quality, and there is no default
configuration change or production deployment based on this experiment.

## Schedule-order confound

The benchmark alternated first/second mode by a globally shuffled case index.
With two repetitions, this put concurrent mode second in both long-prefill
pairs and serial mode second in both decode pairs. Expert-cache warming can
therefore explain apparent schedule effects. These results do not replicate
the claimed benefit of concurrent prefill, which remains experimental.

The same request order ran at both RAM budgets. The table compares the same
schedule and order across those budgets. Future paired benchmarks balance
mode order separately within each workload, as fixed in `a6a07aa`.

Runtime revision: `880011b`. The [measurement record](../bench/results/4090-ram-placement-20260905.json)
contains all requests, startup geometry, both preliminary fidelity runs,
the exact driver, and the retained errors. The original serving checkout
remained unchanged.
