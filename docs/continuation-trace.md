# Exact token continuation diagnostic

`FREETOKEN_CONTINUATION_TRACE_DIR=/absolute/output/directory` enables a diagnostic
capture of the tokens already held on the scheduler's CPU. It records each
existing prefix-match attempt, successful first-chunk admission, and natural
completion before resources are released. Chunked prompts are captured in full.
Admission retries remain attempts until batch preparation succeeds.

The default is off. Disabled hooks add a branch at request boundaries, with no
token conversion, file writes, GPU readbacks, additional cache walks, or per-token
logging. Enabled capture writes and flushes JSONL synchronously and is unsuitable
for a wall-time performance comparison. It changes no routing, cache insertion,
snapshot, request ordering policy, or precision.

Each scheduler process creates its own exclusive file with mode 0600 and a
64 MiB limit. Token IDs expose the prompt and output text to anyone with the
tokenizer. Use synthetic benchmark sessions and archive captures deliberately.
A write, conversion, or budget failure disables capture while serving continues.
Normal scheduler shutdown writes a footer. Missing footers, event gaps,
inconsistent admission lengths, or unfinished admitted requests (including
aborts) make the offline analysis fail. Stop the diagnostic server gracefully
before analysis. A crash is an incomplete diagnostic, not a successful sample.

For one continuing Pi session, extract the numeric suffixes of its ordered
assistant `responseId` values (`chatcmpl-123` maps to uid 123). Do not treat
adjacent requests from different clients or fresh sessions as continuing turns.
Supply that explicit sequence:

```sh
python3 bench/continuation-trace-summary.py /path/to/continuation-PID-ID.jsonl \
  --uids 123,124,125 >continuation-summary.json
```

The summary reports exact common-prefix length, the first differing token,
actual admitted cache reuse, and matching generated tokens that are processed
again. It bounds potential reuse by the recorded consumed length and available
host IDs, excluding an unconsumed final sample and the next prompt's final token
which the existing matcher deliberately excludes. Overlap can advance device
metadata ahead of host output; both lengths are retained separately.

These counts distinguish transcript rewrites from identical tokens being
replayed. They do not prove a usable state snapshot exists at that boundary,
identify the deepest KV-only tree node, or establish saved wall time. Extra
prefix-cache walks would mutate tree structure or recency and are deliberately
absent. For Qwen Flash, deeper continuation must preserve GDN, convolution,
PLE and QSA state together. Attaching later recurrent state to an earlier token
boundary would change the model. Any resulting optimization requires a separate
complete Pi comparison with this diagnostic disabled.
