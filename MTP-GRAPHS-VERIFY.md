# MTP CUDA graph verification

Run these checks on a CUDA system with the Qwen3.8-Flash-Next FTW checkpoint that
contains MTP tensors. Use one shell for the server and a second shell for requests.
Use the production `uring` PLE backend for this graph check. The verify replay copies
the GPU draft token into its persistent pinned host buffer, completes uring staging,
and then submits the captured graph.

Set the checkpoint once in both shells:

```bash
MTP_MODEL=/path/to/qwen3.8-flash-next-ftw
```

## Verify graph enabled

Start the server:

```bash
ft serve --model "$MTP_MODEL" --gpu 0 --max-running-requests 1 --ple-backend uring \
  --cuda-graph-max-bs 1 --speculative-mtp on --mtp-verify-graph on \
  --decode-log-interval 20 2>&1 | tee mtp-graph-on.log
```

Startup must contain all three lines below. The sizes line proves that ordinary
width-one decode was captured, and the final two lines prove that the verify family
contains the required `(bs=1, width=2)` graph.

```text
Start capturing CUDA graphs with sizes: [1]
Start capturing MTP verify CUDA graphs with widths: [2]
Captured MTP verify CUDA graph: bs=1, width=2
```

The final line is the proof that staged uring PLE did not skip the verify graph.
The log must not contain `MTP verify PLE staging failed; using eager verification`.

After the ready message, run one untimed warmup and then three measured 300-token
requests. `--ignore-eos` keeps the output count fixed at 300.

```bash
ft ctl generate "Write a detailed essay about how cities adapt to climate change." \
  --max-tokens 300 --ignore-eos >/dev/null
/usr/bin/time -p ft ctl generate \
  "Write a detailed essay about how cities adapt to climate change." \
  --max-tokens 300 --ignore-eos >mtp-graph-on-1.txt
/usr/bin/time -p ft ctl generate \
  "Write a detailed essay about how cities adapt to climate change." \
  --max-tokens 300 --ignore-eos >mtp-graph-on-2.txt
/usr/bin/time -p ft ctl generate \
  "Write a detailed essay about how cities adapt to climate change." \
  --max-tokens 300 --ignore-eos >mtp-graph-on-3.txt
```

Save the `real` time from each command. Also save the final decode status lines from
`mtp-graph-on.log`, especially `gen throughput (token/s)`, `acceptance rate`, and
`tokens/step`.

## Verify graph disabled

Stop the first server, then start the eager verification control:

```bash
ft serve --model "$MTP_MODEL" --gpu 0 --max-running-requests 1 --ple-backend uring \
  --cuda-graph-max-bs 1 --speculative-mtp on --mtp-verify-graph off \
  --decode-log-interval 20 2>&1 | tee mtp-graph-off.log
```

This run must still log `Start capturing CUDA graphs with sizes: [1]`, and must not
log either MTP verify capture line. Repeat the same warmup and three timed commands,
changing the output filenames to `mtp-graph-off-1.txt` through
`mtp-graph-off-3.txt`.

## MTP disabled

Stop the second server, then start the non-MTP baseline:

```bash
ft serve --model "$MTP_MODEL" --gpu 0 --max-running-requests 1 --ple-backend uring \
  --cuda-graph-max-bs 1 --speculative-mtp off --decode-log-interval 20 \
  2>&1 | tee mtp-disabled.log
```

Repeat the warmup and three timed commands, changing the output filenames to
`mtp-disabled-1.txt` through `mtp-disabled-3.txt`.

## Compare

For each mode, report the median 300-token `real` time and the steady decode
`gen throughput (token/s)`. For both MTP modes, also report the acceptance rate.
Compare the results with the recorded RTX 4090 baselines:

- MTP on with all graphs disabled: 37 to 48 seconds for 300 tokens, with 52 to 78
  percent draft acceptance.
- MTP off: 19 to 31 seconds for 300 tokens.

The primary comparison is MTP on with verify graph on against MTP on with verify
graph off. The MTP-disabled run checks whether accepted drafts now recover enough
launch overhead to beat the ordinary decode baseline.
