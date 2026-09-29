"""Pre-gating accuracy from a FREETOKEN_ROUTE_DUMP directory.

For every DISK layer L (L > 0) and decode step, the actual routes are top-10 of
layer L's router logits; cold routes are those whose expert is not HOT at layer L.
A fetch policy picks F non-HOT experts per layer before layer L runs; coverage is
the share of actual cold routes whose expert was picked (weighted by count).

Predictors:
  pregate   rank non-HOT experts by gate_L(x_{L-1}) (next router on the previous
            layer's router input)
  temporal  the previous step's cold experts at layer L (then pregate order)
  union     temporal first, then pregate
Usage: routedump-analyze.py DUMP_DIR
"""
import glob, json, os, sys
import torch

TOP_K = 10
FETCH = (1, 2, 4, 8, 16)
d = sys.argv[1]
paths = sorted(glob.glob(os.path.join(d, "routes-*.pt")), key=lambda p: (p.rsplit("-", 2)[-2], int(p.rsplit("-", 1)[-1][:-3])))
gates = None
stats = {}  # (predictor, F) -> [covered, total]
cold_per_layer = []
per_layer = {}
for path in paths:
    shard = torch.load(path)
    if "gates" in shard:
        gates = shard["gates"]
    hot = shard["hot_mapping"]
    residency = shard["residency"]
    hidden, logits = shard["hidden"], shard["logits"]
    layers = sorted(hidden)
    for layer in layers:
        if layer == 0 or layer - 1 not in hidden or residency[layer] != "disk":
            continue
        x_prev = hidden[layer - 1]
        steps = min(x_prev.shape[0], logits[layer].shape[0])
        if steps < 2:
            continue
        actual = torch.topk(logits[layer][:steps], TOP_K, dim=-1).indices  # [S, 10]
        is_hot = torch.zeros(logits[layer].shape[-1], dtype=torch.bool)
        if hot is not None:
            row = hot[layer] if hot.dim() == 2 else hot
            is_hot = row >= 0
        pre = x_prev[:steps] @ gates[layer].T  # [S, E]
        pre[:, is_hot] = float("-inf")
        order = torch.argsort(pre, dim=-1, descending=True)
        for s in range(1, steps):
            cold = [int(e) for e in actual[s] if not is_hot[e]]
            if not cold:
                continue
            cold_per_layer.append(len(cold))
            prev_cold = [int(e) for e in actual[s - 1] if not is_hot[e]]
            ranked = [int(e) for e in order[s, : max(FETCH) * 2]]
            union = list(dict.fromkeys(prev_cold + ranked))
            for name, ranking in (("pregate", ranked), ("temporal", list(dict.fromkeys(prev_cold + ranked))), ("union", union)):
                for f in FETCH:
                    picked = set(ranking[:f]) if name != "temporal" else set(prev_cold[:f])
                    covered = sum(1 for e in cold if e in picked)
                    st = stats.setdefault((name, f), [0, 0])
                    st[0] += covered
                    st[1] += len(cold)
            pl = per_layer.setdefault(layer, [0, 0, 0])
            pl[0] += len(cold)
            pl[1] += sum(1 for e in cold if e in set(ranked[:4]))
            pl[2] += 1

print(f"shards={len(paths)} layer-steps with cold routes={len(cold_per_layer)} "
      f"mean cold routes per DISK layer-step={sum(cold_per_layer)/max(1,len(cold_per_layer)):.2f}")
for name in ("pregate", "temporal", "union"):
    row = {f: round(stats[(name, f)][0] / max(1, stats[(name, f)][1]), 3) for f in FETCH if (name, f) in stats}
    print(name, "coverage by fetch budget F:", row)
print("per-layer pregate coverage at F=4:", {l: round(v[1] / max(1, v[0]), 2) for l, v in sorted(per_layer.items())})
