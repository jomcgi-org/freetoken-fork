"""Causal CPU proposals. Every returned token still requires target verification."""

from array import array


WIDTH = 5
MATCH = 8
LOOKBACK = 8192


def propose(tokens, *, match=MATCH, drafts=WIDTH - 1, lookback=LOOKBACK):
    """Copy a prior continuation of the current suffix from known tokens only.

    Prefer the most recent complete match. Byte search keeps a long prompt out
    of the Python inner loop; alignment checks exclude partial integer matches.
    Overlapping occurrences are allowed only when the entire draft is known.
    """
    if match < 1 or drafts < 1 or lookback < match + drafts:
        raise ValueError("invalid ngram proposal geometry")
    recent = tokens[-lookback:]
    if len(recent) < match + drafts:
        return None
    values = array("I", recent)
    unit = values.itemsize
    data = values.tobytes()
    suffix = data[-match * unit:]
    end = len(data) - drafts * unit
    while end >= len(suffix):
        found = data.rfind(suffix, 0, end)
        if found < 0:
            return None
        if found % unit == 0:
            start = found // unit + match
            return list(values[start:start + drafts])
        end = found + len(suffix) - 1
    return None


def proposal_for_request(req):
    if (req.remain_len < WIDTH or req.cached_len + 1 != req.device_len
            or req.input_ids.numel() != req.device_len
            or not req.sampling_params.is_greedy
            or req.sampling_params.guided_decoding is not None
            or getattr(req, "guided_state", None) is not None
            or getattr(req, "mm_embeds", None) is not None):
        return None
    lazy = getattr(req, "lazy_kv_restore", None)
    if lazy is not None and not lazy.complete:
        return None
    # Let the ordinary path process and snapshot a newly emitted tool opener.
    anchor = getattr(req, "toolcall_anchor_len", None)
    if anchor is not None and req.cached_len <= anchor:
        return None
    return propose(req.input_ids[-LOOKBACK:].tolist())
