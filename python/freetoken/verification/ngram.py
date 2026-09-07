"""Causal CPU proposals. Every returned token still requires target verification."""

from array import array


WIDTH = 5
MATCH = 8
LOOKBACK = 8192


def note_verification(req, matched):
    """Pause weak drafts without changing the target's output distribution."""
    if not 0 <= matched < WIDTH:
        raise ValueError("invalid ngram acceptance length")
    if matched < 2:
        weak = min(getattr(req, "_ngram_weak_windows", 0) + 1, 5)
        req._ngram_weak_windows = weak
        req._ngram_retry_at = req.device_len + (16 << (weak - 1))
    else:
        req._ngram_weak_windows = 0
        req._ngram_retry_at = 0


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


def pending_proposal_possible(tokens):
    """Check a necessary condition without reading the pending sampled token.

    Any full match must contain the known MATCH-1 suffix. Only WIDTH-1 known
    following tokens are required: the pending token may also be the last draft.
    A positive result still needs the normal lookup after the host-copy fence.
    """
    return propose(tokens, match=MATCH - 1, drafts=WIDTH - 1,
                   lookback=LOOKBACK - 1) is not None


def proposal_eligible(req, *, pending_tokens=0):
    """Check request constraints before waiting for an overlapped host token."""
    if pending_tokens not in (0, 1):
        raise ValueError("ngram lookup supports at most one pending token")
    if (req.device_len < getattr(req, "_ngram_retry_at", 0)
            or req.remain_len < WIDTH or req.cached_len + 1 != req.device_len
            or req.input_ids.numel() + pending_tokens != req.device_len
            or not req.sampling_params.is_greedy
            or req.sampling_params.guided_decoding is not None
            or getattr(req, "guided_state", None) is not None
            or getattr(req, "mm_embeds", None) is not None):
        return False
    lazy = getattr(req, "lazy_kv_restore", None)
    if lazy is not None and not lazy.complete:
        return False
    # Let the ordinary path process and snapshot a newly emitted tool opener.
    anchor = getattr(req, "toolcall_anchor_len", None)
    if anchor is not None and req.cached_len <= anchor:
        return False
    return True


def proposal_for_request(req, *, pending_token=None):
    """Inspect causal history without committing a pending sampled host token.

    The caller must fence the host copy before supplying pending_token. A lookup
    is advisory: drain prior output and recheck eligibility before verification.
    """
    pending = int(pending_token is not None)
    if not proposal_eligible(req, pending_tokens=pending):
        return None
    tokens = req.input_ids[-(LOOKBACK - pending):].tolist()
    if pending:
        tokens.append(int(pending_token))
    return propose(tokens)
