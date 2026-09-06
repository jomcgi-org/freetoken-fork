"""First-token state retention for the explicit target-verification diagnostic.

Never imported by normal serving. Capture copies become operations in a separate
verification graph; they are part of its measured cost, including acceptance.
"""

from contextlib import contextmanager


_ACTIVE = None


class SeedCheckpoint:
    def __init__(self, views, *, gdn_sources, ple_layers, qsa_layers, width=2,
                 retain_recurrent=True):
        import torch

        required = {"conv", "recurrent", "slot/ple_conv", "slot/ple_ngram_ctx", "qsa_pending"}
        if set(views) != required:
            raise ValueError("checkpoint requires exactly the supported Qwen request state")
        if width not in (2, 3, 5):
            raise ValueError("checkpoint width must be two, three or five")
        self.width = width
        self.views = views
        self.prefixes = [{name: torch.empty_like(value) for name, value in views.items()
                          if retain_recurrent or name != "recurrent"}
                         for _ in range(width - 1)]
        self.saved = self.prefixes[0]
        self.gdn_sources, self.ple_layers, self.qsa_layers = gdn_sources, ple_layers, qsa_layers
        self.expected = {("gdn", i): width for i in range(views["recurrent"].shape[0])}
        self.expected.update({("ple", i): width for i in range(views["slot/ple_conv"].shape[0])})
        self.expected.update({("qsa", i): width for i in range(views["qsa_pending"].shape[0])})
        self.expected[("ngram", 0)] = 1
        for mapping, name in ((gdn_sources, "recurrent"), (ple_layers, "slot/ple_conv"),
                              (qsa_layers, "qsa_pending")):
            if set(mapping.values()) != set(range(views[name].shape[0])):
                raise ValueError("checkpoint layer mapping does not cover its state")
        self.counts = {}
        self.ready = False

    @classmethod
    def from_engine(cls, engine, req, views, *, width=2):
        pool = engine.linear_state_pool
        return cls(views, width=width,
                   gdn_sources={pool.recurrent_states[i].data_ptr(): i
                                for i in range(pool.recurrent_states.shape[0])},
                   ple_layers=dict(pool._state_layer_index["ple_conv"]),
                   qsa_layers=dict(engine.attn_backend._idx_slot))

    def begin(self):
        self.counts = {}
        self.ready = False

    def _visit(self, kind, index):
        key = (kind, index)
        count = self.counts.get(key, 0)
        if key not in self.expected or count >= self.expected[key]:
            raise RuntimeError("unexpected checkpoint state update")
        self.counts[key] = count + 1
        return count

    def capture_gdn(self, state_source, *, args=None, kwargs=None, update=None):
        index = self.gdn_sources[state_source.data_ptr()]
        step = self._visit("gdn", index)
        if step < self.width - 1:
            for name in ("conv", "recurrent"):
                self.prefixes[step][name][index].copy_(self.views[name][index])

    def capture_ple(self, layer_id):
        index = self.ple_layers[layer_id]
        step = self._visit("ple", index)
        if step < self.width - 1:
            self.prefixes[step]["slot/ple_conv"][index].copy_(self.views["slot/ple_conv"][index])

    def capture_qsa(self, layer_id):
        index = self.qsa_layers[layer_id]
        step = self._visit("qsa", index)
        if step < self.width - 1:
            self.prefixes[step]["qsa_pending"][index].copy_(self.views["qsa_pending"][index])

    def capture_ngram(self, meta):
        import torch

        destination = self.saved["slot/ple_ngram_ctx"]
        if (meta.input_ids.numel() != self.width or destination.ndim != 2
                or destination.shape[0] != 1 or destination.shape[1] < 1
                or meta.ngram_context.shape != destination.shape):
            raise ValueError("checkpoint requires one matching-width ngram history")
        self._visit("ngram", 0)
        # Same integer shift as ordinary width-one commit; the live context still
        # receives the original two-token commit after this copy.
        for length, prefix in enumerate(self.prefixes, 1):
            history = torch.cat((meta.ngram_context, meta.input_ids[:length].reshape(1, -1)), dim=1)
            prefix["slot/ple_ngram_ctx"].copy_(history[:, -destination.shape[1]:])

    def finish(self):
        if self.counts != self.expected:
            raise RuntimeError("checkpoint did not observe every expected state update")
        self.ready = True

    def restore(self, prefix_len=1):
        if not self.ready:
            raise RuntimeError("checkpoint is not ready")
        if not 1 <= prefix_len < self.width:
            raise ValueError("checkpoint prefix is outside the retained range")
        for name, value in self.views.items():
            value.copy_(self.prefixes[prefix_len - 1][name])

    def owned_tensor_bytes(self):
        return sum(value.numel() * value.element_size()
                   for prefix in self.prefixes for value in prefix.values())


@contextmanager
def capture_context(checkpoint):
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("nested checkpoint capture")
    _ACTIVE = checkpoint
    try:
        yield
    finally:
        _ACTIVE = None


def install():
    """Wrap established operations without changing their arguments or outputs."""
    from freetoken.models.qwen4_exp import gdn, model, ple
    from freetoken.attention.qsa_sparse import QSASparseAttnBackend

    original_model = model.Qwen4ExpModel.forward
    original_gdn = gdn.gdn_decode_fla
    original_ple = ple.PLELayer._decode_conv
    original_qsa = QSASparseAttnBackend._qsa_forward_one
    original_ngram = ple.commit_ngram_context

    def forward(network, input_ids, batch):
        checkpoint = _ACTIVE
        if checkpoint is None:
            return original_model(network, input_ids, batch)
        if not getattr(batch, "mtp_fused", False) or input_ids.numel() != checkpoint.width:
            raise RuntimeError("checkpoint capture requires a matching-width fused forward")
        checkpoint.begin()
        result = original_model(network, input_ids, batch)
        checkpoint.finish()
        return result

    def recurrent(*args, **kwargs):
        result = original_gdn(*args, **kwargs)
        if _ACTIVE is not None:
            _ACTIVE.capture_gdn(kwargs["state_source"], args=args, kwargs=kwargs, update=original_gdn)
        return result

    def conv(layer, x, meta, states):
        result = original_ple(layer, x, meta, states)
        if _ACTIVE is not None:
            _ACTIVE.capture_ple(layer.layer_id)
        return result

    def sparse(backend, q, k, v, index, layer_id, batch):
        result = original_qsa(backend, q, k, v, index, layer_id, batch)
        if _ACTIVE is not None:
            _ACTIVE.capture_qsa(layer_id)
        return result

    def ngram(meta, fla, context_pool=None):
        if _ACTIVE is not None:
            _ACTIVE.capture_ngram(meta)
        return original_ngram(meta, fla, context_pool)

    model.Qwen4ExpModel.forward = forward
    gdn.gdn_decode_fla = recurrent
    ple.PLELayer._decode_conv = conv
    QSASparseAttnBackend._qsa_forward_one = sparse
    ple.commit_ngram_context = ngram
