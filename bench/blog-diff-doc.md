# Blog diff: what changed after the 2026-09-01 post

Companion to `docs/posts/2026-09-01-125b-on-a-4090.md` in the homelab repo. That
post describes the fork as it stood on 1 September. This records what the two
rounds of work after it established: the numbers that moved, the claims in the
post that are now wrong or incomplete, and what any of it is expected to mean
for GLM on a cloud box.

Evidence for everything here is in `bench/RESULTS.md`. Where a number came from
a single arm rather than a controlled pair, that is stated.

## 1. Headline numbers

| | blog (01 Sep) | now | note |
|---|---:|---:|---|
| single stream, warm | 21 | 22 to 24 | deployed: `--moe-cpu-willneed recent`, `--moe-hot-adapt-prefill-weight 0.1` |
| post-document decode (76k prompt, then 200-token turns) | not measured | 12.5 to 18.3 | `--moe-hot-adapt-histories split`, measured, not yet deployed |
| prefill, 441-token warm | 116 | unchanged | not revisited |
| eight streams, aggregate | 37 | unchanged | not revisited |

The single-stream figure is the one the post leads with, and it moved about 10
percent. The larger result of the second and third rounds is not in that row at
all: it is a workload the post never measured, a long document followed by
conversation, where decode improved about 30 percent.

## 2. Corrections to the post

### 2.1 Speculative decode: the stated reason is wrong

Section 6 lists speculative decode as a loss and explains it as "speculation
pays when a marginal token is cheap. With part of every step on the CPU tier a
drafted token costs full price, even at 72% acceptance; 12.6 vs 18.0 tok/s".

That explanation is not what the measurement shows. MTP forces CUDA graphs off
entirely (`engine.py` built the graph runner with an empty `cuda_graph_bs`
whenever `speculative_mtp == "on"`), so every verify forward paid full eager
launch overhead. Acceptance was never the problem: it measured 52 to 78 percent,
which is comfortably enough to pay. The cost was structural and fixable rather
than intrinsic to the tier.

The fork now captures a width-keyed verify graph alongside the width-1 decode
graphs. Whether that turns the loss into a win is being measured; the honest
status is "the stated reason was wrong and the real one is addressed", not "this
is now a win".

### 2.2 The 1,000-step adaptation interval makes short experiments blind

Section 5.5 documents the hot-set adapter re-ranking its counters every 1,000
steps. That is a reasonable production choice, and production does converge:
long-running it sits within about five points of the hindsight-optimal hot set
(oracle 84 to 97 percent against realised 78 to 93).

It also means no short experiment can measure anything that changes what the
adapter aims at. Every A/B arm here restored a hot plan adapted to production
traffic and then ran 600 to 1800 decode steps, so the adapter ticked zero or one
times. Oracle versus realised was 92 to 96 against 59 to 69 percent in an essay
arm and 87 to 95 against 43 to 49 after a long document: the arms were measuring
a stale, mismatched plan rather than the knob under test.

To be clear about scope, since this is easy to overstate: none of the five
entries in the post's "didn't pan out" table are adaptation knobs. They are the
PCIe cold-tail paths, the userspace pager, router-ahead prefetch and speculative
decode, and none of them depend on the adapter running. The table stands. The
warning applies to the rounds of work after the post, where three knobs read
neutral for this reason alone, and to anyone who tries to reproduce this kind of
result with a short benchmark.


### 2.3 The essay benchmark cannot resolve a small effect

An ABBA-ordered pair (control, knob, knob, control, so page-cache warming
cancels) produced arm medians of 23.8, 29.4, 25.9 and 29.1 tok/s, with positions
two and three being the *same* arm. Neither the knob nor a warming trend
explains that pattern. The noise floor on that benchmark is therefore somewhere
around 10 to 20 percent, and several rounds of knobs were contesting 1 to 5
percent. "Neutral" in this work means "below the noise floor", not "proven
zero".

## 3. What is in the fork but not in the post

### 3.1 Split prefill and decode routing histories

The one clear win of the later rounds, and the post predates it.

A long prefill floods the shared decayed routing counters, so the hot set
re-aims at a distribution the following decode never uses. Keeping separate
prefill and decode histories fixes that. Measured as a full 2x2, both arms in
both positions:

| metric | shared (pos1 / pos2) | split (pos1 / pos2) |
|---|---|---|
| realised hot_pair_rate | 47.0% / 48.0% | 67.9% / 67.2% |
| decode batches, mean of 9 | 19.37 / 19.34 | 23.87 / 26.37 |
| post-document turns | 12.5 / 15.5 tok/s | 19.2 / 17.4 tok/s |
| long prompt prefill wall | 664 s / 648 s | 817 s / 885 s |

Shared measures 19.37 and 19.34 decode batches in opposite positions and split's
pair rate lands at 67.9 and 67.2, so position accounts for none of it.

The cost is real: prefill is about 30 percent slower. It is a decode-versus-
prefill trade, right for reading a document once and discussing it over many
turns, wrong for prefilling a large input to emit a short answer.

### 3.2 Measured neutral, kept behind flags

- **Unequal per-layer hot capacity.** The planner's own estimate caps the gain
  at half a point of pair rate (`profiled hot_pair_rate equal=73.9%,
  chosen=74.4%`) and the measurement agreed. Capacity was genuinely
  redistributed (59/82/104 rows against a flat 82), so the mechanism works and
  this model simply has near-uniform routing concentration across its 28 DISK
  layers. Worth retesting on a model whose routing is uneven.
- **Skipping the CPU handoff when every routed expert is already hot.** All-hot
  layers measured 2.9 to 3.5 of 28, bounding the saving at a fraction of a
  millisecond against 16 to 19 ms of CPU windows.
- **PINNED hot set, decode worker thread policy, spin versus hybrid barrier.**
  All neutral. The barrier result is worth keeping in mind: 83 percent of
  task-body self time sits in the pass barrier spin, but replacing it with a
  condvar wake costs more than the spin it removes, and the spinners occupy
  sibling cores that would otherwise idle.

### 3.3 Fixes with no user-visible number

- Host memory governor now respects cgroup limits, taking the minimum of the
  host and cgroup ceilings. It subtracts reclaimable page cache from the cgroup
  usage, which matters enormously here because the disk expert tier works by
  keeping the page cache full: the naive `limit - current` reads a box with 40
  GiB of cache as having 1 GiB of headroom instead of 41.
- Upstream sync, and a QUICKSTART correction naming uring as the PLE backend of
  record after the HMM backend was retired for UVM oopses.

## 4. Method notes worth publishing

These cost more time than most of the optimisations and generalise beyond this
project.

1. **Verify the mechanism fired before believing a null result.** Three separate
   false neutrals this round: the adapter never ticked; a prefill-history
   normalisation divided the history weight by the token count and made a knob a
   no-op by construction; and a test fix that looked correct changed nothing.
2. **Make the control prove the failure.** The stream-leak fix for an
   order-dependent test was verified by running the broken tree first and
   requiring it to fail. It did not fail differently from the fixed tree, which
   is the only reason a wrong fix was not shipped.
3. **Judge a hot-set change by pair rate, not throughput.** Pair rate is a
   property of the hot set against observed routing and is order-independent;
   throughput on this box is dominated by page-cache position, worth roughly 20
   percent between arm positions.
4. **Defeat the prefix cache in any prefill experiment.** A long-document arm
   that returns in 11 seconds instead of 650 hit the KV disk cache and measured
   nothing.

## 5. What this predicts for GLM on a cloud box

GLM 5.3 Flash NVFP4 is about 160 GB of expert banks (177 GiB FTW), 45 layers,
288 routed plus 1 shared expert, top-8. The only sensible single-GPU rung is
`g4-standard-48`: 48 vCPU, 180 GB RAM, one full RTX PRO 6000 96 GB, about
$4.50/hour. The smaller G4 types are fractional vGPU slices (`g4-standard-6` is
one eighth of a card), so they cannot hold the 48 GB hot set the working rung
used, and the next rung up doubles the GPU count to buy RAM.

Measured there previously: 257 tok/s prefill, 11 to 12 tok/s single-stream
decode, 19.3 aggregate at four streams.

### 5.1 The decode number is a method artefact, not a hardware limit

With 112 GB pinned, 29 layers were PINNED and streamed experts over PCIe every
step. The arithmetic: 160 GB over 12,960 experts is about 12.3 MB each, top-8
across 45 layers is 360 activations per token, and a 48 GB hot set covers about
30 percent, so roughly 250 misses cross the bus per token, about 3.1 GB. At
realistic PCIe throughput that is 60 to 125 ms per token, and the measured step
was about 70 ms. Decode was bus-bound.

That is why the 4090 gets 22 to 24 tok/s at about 10 percent hot coverage while
GLM got 11 to 12 at 30 percent: the 4090 never ships experts to the GPU for DISK
layers. It reads them from page cache, computes them on the CPU, and returns a
small activation vector, kilobytes instead of megabytes.

**The single most valuable untested configuration is GLM on the CPU-executor
path.** It is fork issue #12. It was attempted and the server was SIGKILLed
about ten seconds after start, twice, with nothing in dmesg, which is the
signature of a cgroup kill and therefore plausibly the same bug the memory
governor fix addresses.

### 5.2 Transfer expectations, ranked

| technique | expected transfer | reasoning |
|---|---|---|
| cgroup-aware memory governor | direct, and defensive | the 64 GB-capped rung thrashed at 0.03 tok/s and 140,835 major faults per step; this is that failure mode |
| CPU executor for the cold tail | large if it runs at all | removes 3.1 GB per token of bus traffic; 48 vCPU against the 4090 box's 8 cores |
| split prefill and decode histories | scales with hot-set scarcity | +30 percent here; GLM's decode is the weak metric at 11 to 12 tok/s |
| unequal per-layer hot capacity | worth retesting, may behave differently | neutral here only because routing is near-uniform; 288 routed plus 1 shared with sigmoid routing is the plausible counter-example |
| MTP verify graphs | idea transfers, code partly does not | GLM uses KDA linear attention rather than the FLA path made capturable here, and needs its own draft head |
| `--moe-cpu-willneed recent` | only in a disk-backed configuration | it schedules page prefetch; no faults, no benefit |

### 5.3 Cost, and where the approach stops making sense

Offloading converts a capacity problem into a bandwidth problem, and bandwidth
is paid per token rather than per model. So it is strong exactly where the GPU
would otherwise idle and weak where you want to saturate it.

The measured aggregate scaling says so directly: 11 to 12 tok/s at one stream
against 19.3 aggregate at four, about 1.68x for 4x the streams. With top-8 of
288 experts, four concurrent streams touch up to 32 distinct experts per layer
instead of 8, so batching fetches nearly four times the bytes for four times the
tokens. There is no amortisation path.

| regime | single GPU, offloaded | multi-GPU, HBM-resident |
|---|---|---|
| one stream | about $104 per million tokens, or about $22 on spot | roughly $244 per million tokens |
| high concurrency | no meaningful improvement | roughly $15 per million tokens |

The single-GPU numbers are measured; the multi-GPU column is an estimate from
list pricing and assumed throughput, and no HBM-resident GLM baseline has been
run. It is worth running one, precisely so the single-GPU claim has something
honest to be compared against.

The conclusion the post should carry: this technique is for one to a few
streams on hardware you already have, where the alternative is not running the
model at all. It is not a serving architecture.
