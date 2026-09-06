"""Independent cumulative checks, kept outside the agent's writable fixture."""

import argparse
import importlib.util
import json
from pathlib import Path


def check(cache_type, stage):
    c = cache_type()
    c.put("a", 7, 5, 10)
    assert c.get("a", 14.999) == 7, "entry must survive before expiry"
    assert c.get("a", 15, "missing") == "missing", "expiry boundary is inclusive"
    for ttl in (0, -2):
        c.put("a", 7, 20, 10)
        c.put("a", 8, ttl, 11)
        assert c.get("a", 11, False) is False, "nonpositive TTL invalidates key"
    c.put("none", None, 20, 10)
    assert c.get("none", 11, "missing") is None, "None is a stored value"
    assert c.get("absent", 11, 0) == 0
    if stage == 1:
        return
    for capacity in (0, -1):
        try:
            cache_type(capacity=capacity)
        except ValueError:
            pass
        else:
            raise AssertionError("nonpositive capacity must raise ValueError")
    c = cache_type(capacity=2)
    c.put("a", 1, 100, 0)
    c.put("b", 2, 100, 0)
    assert c.get("a", 1) == 1
    c.get("absent", 1)
    c.put("c", 3, 100, 1)
    assert c.get("b", 2) is None, "get must promote a, so b is evicted"
    c.put("a", 10, 100, 2)
    assert c.get("c", 2) == 3, "updating a must not evict c"
    c.put("d", 4, 100, 3)
    assert c.get("a", 3) is None and c.get("c", 3) == 3
    c = cache_type(capacity=2)
    c.put("a", 1, 100, 0)
    c.put("b", 2, 100, 0)
    c.put("a", 10, 100, 1)
    c.put("c", 3, 100, 1)
    assert c.get("a", 1) == 10 and c.get("b", 1) is None, "put must promote key"
    c = cache_type(capacity=2)
    c.put("live", 1, 100, 0)
    c.put("expired", 2, 1, 0)
    c.put("new", 3, 100, 2)
    assert c.get("live", 2) == 1, "purge expired entries before evicting live LRU"
    assert c.get("new", 2) == 3
    c = cache_type(capacity=None)
    for i in range(9):
        c.put(i, i, 100, 0)
    assert all(c.get(i, 1) == i for i in range(9)), "default capacity is unbounded"
    if stage == 2:
        return
    c = cache_type(capacity=2)
    calls = []

    def loader():
        calls.append(1)
        return None

    assert c.get_or_load("n", loader, 2, 0) is None
    assert c.get_or_load("n", loader, 2, 1) is None
    assert len(calls) == 1, "cached None must bypass loader"
    c.get_or_load("n", loader, 2, 2)
    assert len(calls) == 2, "expiry must call loader once"
    c.put("other", 10, 100, 2)
    c.get_or_load("n", loader, 2, 3)
    c.put("new", 20, 100, 3)
    assert c.get("other", 3) is None, "loader cache hit must promote key"
    failure = RuntimeError("sentinel loader failure")

    def broken():
        raise failure

    try:
        c.get_or_load("bad", broken, 5, 3)
    except RuntimeError as exc:
        assert exc is failure, "propagate the same loader exception"
    else:
        raise AssertionError("loader error must propagate")
    assert c.get("new", 3) == 20 and c.get("n", 3, "missing") is None
    c.get_or_load("zero", lambda: 9, 0, 3)
    assert c.get("zero", 3, "missing") == "missing"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("stage", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    try:
        spec = importlib.util.spec_from_file_location("candidate", args.workspace / "cache.py")
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        check(candidate.Cache, args.stage)
    except Exception as exc:
        print(json.dumps(dict(passed=False, stage=args.stage, error=f"{type(exc).__name__}: {exc}")))
        return 1
    print(json.dumps(dict(passed=True, stage=args.stage)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
