"""Multi-token delivery must produce the same text as serial token delivery."""

import pytest

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


class ByteTokenizer:
    eos_token_id = 256

    def __init__(self):
        self.calls = 0

    def batch_decode(self, rows):
        self.calls += 1
        return [bytes(row).decode("utf-8", errors="replace") for row in rows]


def messages(text, uid=1, *, stop=None):
    result = [DetokenizeMsg(uid=uid, next_token=token, finished=False, stop_strs=[stop] if stop else None)
              for token in text.encode()]
    if stop:
        result[-1].finished = True
        result[-1].matched_stop = stop
    else:
        result.append(DetokenizeMsg(uid=uid, next_token=256, finished=True))
    return result


@pytest.mark.parametrize("width", [2, 3, 5, 17])
@pytest.mark.parametrize("text", ["alpha bravo charlie\n", "a 👩‍💻 中文 café\n"])
def test_repeated_uid_matches_serial_decoding_across_chunk_boundaries(width, text):
    stream = messages(text)
    serial = DetokenizeManager(ByteTokenizer())
    expected = [serial.detokenize([message])[0] for message in stream]
    batched = DetokenizeManager(ByteTokenizer())
    actual = []
    for start in range(0, len(stream), width):
        actual.extend(batched.detokenize(stream[start:start + width]))
    assert actual == expected
    assert "".join(actual) == text
    assert not batched.decode_map


def test_interleaved_requests_preserve_per_message_order_and_terminal_cleanup():
    first, second = messages("alpha\n", 1), messages("中\n", 2)
    stream = []
    while first or second:
        stream.extend(first[:2]); del first[:2]
        stream.extend(second[:1]); del second[:1]
    serial = DetokenizeManager(ByteTokenizer())
    expected = [serial.detokenize([message])[0] for message in stream]
    batched = DetokenizeManager(ByteTokenizer())
    assert batched.detokenize(stream) == expected
    assert not batched.decode_map


def test_stop_prefix_never_leaks_from_a_multi_token_batch():
    stream = messages("prefix STOP", stop="STOP")
    serial = DetokenizeManager(ByteTokenizer())
    expected = [serial.detokenize([message])[0] for message in stream]
    batched = DetokenizeManager(ByteTokenizer())
    actual = batched.detokenize(stream)
    assert actual == expected
    assert "".join(actual) == "prefix "
    assert not batched.decode_map


def test_distinct_requests_keep_batched_tokenizer_calls():
    tokenizer = ByteTokenizer()
    manager = DetokenizeManager(tokenizer)
    stream = [messages("a", 1)[0], messages("b", 2)[0], messages("c", 3)[0]]
    assert manager.detokenize(stream) == ["a", "b", "c"]
    assert tokenizer.calls == 2
