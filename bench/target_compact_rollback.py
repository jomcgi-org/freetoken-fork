"""Compact recurrent rollback for the explicitly installed verification probe."""


def make_checkpoint_type(base_type):
    class CompactRollback(base_type):
        def __init__(self, views, **kwargs):
            import torch

            super().__init__(views, retain_recurrent=False, **kwargs)
            self.initial_recurrent = torch.empty_like(views["recurrent"])
            self.updates = {}
            self.restore_graphs = {}

        def begin(self):
            super().begin()
            self.initial_recurrent.copy_(self.views["recurrent"])

        def capture_gdn(self, state_source, *, args=None, kwargs=None, update=None):
            import torch

            index = self.gdn_sources[state_source.data_ptr()]
            step = self._visit("gdn", index)
            if step == self.width - 1:
                return  # Fully accepted verification keeps its live final state.
            expected_keys = {"A_log", "dt_bias", "state_source", "indices", "cu_seqlens", "scale"}
            if (args is None or len(args) != 5 or not all(isinstance(x, torch.Tensor) for x in args)
                    or kwargs is None or set(kwargs) != expected_keys or not callable(update)):
                raise ValueError("compact rollback requires the established GDN update signature")
            key = (index, step)
            if key not in self.updates:
                if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("recurrent inputs must be provisioned before graph capture")
                self.updates[key] = dict(args=tuple(torch.empty_like(x) for x in args),
                                         kwargs=dict(kwargs), update=update)
            saved = self.updates[key]
            if saved["update"] is not update:
                raise RuntimeError("recurrent update function changed")
            for name, actual in kwargs.items():
                prior = saved["kwargs"][name]
                if isinstance(actual, torch.Tensor):
                    same = (isinstance(prior, torch.Tensor) and prior.data_ptr() == actual.data_ptr()
                            and prior.shape == actual.shape and prior.dtype == actual.dtype
                            and prior.stride() == actual.stride() and prior.device == actual.device)
                else:
                    same = prior == actual
                if not same:
                    raise RuntimeError("recurrent update binding changed: " + name)
            for destination, actual in zip(saved["args"], args):
                if (destination.shape != actual.shape or destination.dtype != actual.dtype
                        or destination.device != actual.device):
                    raise RuntimeError("recurrent input geometry changed")
                destination.copy_(actual)
            self.prefixes[step]["conv"][index].copy_(self.views["conv"][index])

        def finish(self):
            super().finish()
            if len(self.updates) != len(self.gdn_sources) * (self.width - 1):
                self.ready = False
                raise RuntimeError("recurrent input history is incomplete")

        def replay_eager(self, prefix_len):
            if not self.ready:
                raise RuntimeError("checkpoint is not ready")
            if not 1 <= prefix_len < self.width:
                raise ValueError("checkpoint prefix is outside the retained range")
            self.views["recurrent"].copy_(self.initial_recurrent)
            for index in range(len(self.gdn_sources)):
                for step in range(prefix_len):
                    call = self.updates[(index, step)]
                    call["update"](*call["args"], **call["kwargs"])
            for name, value in self.prefixes[prefix_len - 1].items():
                self.views[name].copy_(value)

        def capture_restore_graphs(self, stream):
            import torch

            if self.restore_graphs:
                raise RuntimeError("rollback graphs were already captured")
            for prefix_len in range(1, self.width):
                self.replay_eager(prefix_len)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self.replay_eager(prefix_len)
                self.restore_graphs[prefix_len] = graph
            stream.synchronize()

        def restore(self, prefix_len=1):
            if not self.ready:
                raise RuntimeError("checkpoint is not ready")
            if prefix_len not in self.restore_graphs:
                raise RuntimeError("requested rollback graph is unavailable")
            self.restore_graphs[prefix_len].replay()

        def owned_tensor_bytes(self):
            tensors = [self.initial_recurrent]
            tensors += [value for call in self.updates.values() for value in call["args"]]
            return super().owned_tensor_bytes() + sum(x.numel() * x.element_size() for x in tensors)

        def close(self, stream):
            stream.synchronize()
            for graph in self.restore_graphs.values():
                graph.reset()
            self.restore_graphs.clear()

    return CompactRollback
