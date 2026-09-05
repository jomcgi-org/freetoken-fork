"""Short greedy prompt-fidelity checks when changing expert execution placement.

These checks catch lost prompt content and basic reasoning regressions. They
complement numerical parity and matched-output checks; they are not a broad
model-quality evaluation. Thinking is disabled explicitly so the answer budget
cannot be consumed by hidden reasoning.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def cases():
    records = [f"Record {i:03}: package parcel{i:03} has status queued." for i in range(120)]
    records[51] = "Record 051: package parcel051 has status delivered; its receipt is VIOLET-68243."
    return [
        ("arithmetic", "Compute 17 + 28 + 45 + 96 + 133. Output only the integer result.", "319"),
        ("codeword", "Remember the access code ZEPHYR-58319. Return that exact access code and nothing else.", "ZEPHYR-58319"),
        ("position", "The list is [amber, birch, cobalt, delta, elm]. Output only its third item.", "cobalt"),
        ("code_trace", "In Python, what is sum(n*n for n in [2,5,8] if n % 2 == 0)? Output only the integer.", "68"),
        ("negation", "A farmer has 17 sheep. All but 9 run away. How many remain? Output only the integer.", "9"),
        ("ordering", "Repeat these IDs in their original order, separated by single spaces, with no other text: Z5 A12 C3 B8", "Z5 A12 C3 B8"),
        ("context_override", "In this fictional ledger, the capital listed for Norway is Bergen, not Oslo. According to this ledger, what is the capital? Output only the city name.", "Bergen"),
        ("long_retrieval", "Read these shipping records:\n" + "\n".join(records) + "\nWhat is the receipt for parcel051? Return only the exact receipt code.", "VIOLET-68243"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18090")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failures = 0
    with args.output.open("w") as out:
        for name, prompt, expected in cases():
            payload = {
                "model": args.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(
                args.base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
            answer = result["choices"][0]["message"].get("content") or ""
            passed = answer.strip() == expected
            failures += not passed
            row = dict(
                name=name, prompt=prompt, expected=expected, answer=answer,
                passed=passed, wall_s=time.perf_counter() - start,
                usage=result.get("usage"),
            )
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({k: v for k, v in row.items() if k != "prompt"}), flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
