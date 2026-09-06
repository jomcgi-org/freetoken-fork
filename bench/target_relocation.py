"""Borrowed request slots and page relocation for an isolated diagnostic worker.

Never used by serving. The caller must abort the diagnostic request afterward:
reference forwards also change expert-cache history and adaptation counters.
"""

import copy


def eligible_boundary(position, page_size, width, remaining):
    return (width in (3, 5) and page_size >= width + 3 and position >= 0
            and position % page_size == page_size - 2 and remaining >= width + 3)


def linear_slot(req):
    return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx


def request_plan(source_table, source_linear, spare_table, spare_linear):
    return ((source_table, source_linear), (spare_table, source_linear),
            (source_table, spare_linear), (spare_table, spare_linear))


def qualify(records, layout, width):
    if not isinstance(layout, dict):
        return False
    required = {"boundary", "source_table", "source_linear", "spare_table", "spare_linear",
                "old_page", "reserved_page", "page_size"}
    if set(layout) != required:
        return False
    if (layout["source_table"] == layout["spare_table"]
            or layout["source_linear"] == layout["spare_linear"]
            or layout["old_page"] == layout["reserved_page"]
            or layout["old_page"] == layout["reserved_page"] + 1):
        return False
    boundary = layout["boundary"]
    if not eligible_boundary(boundary - 2, layout["page_size"], width, width + 3):
        return False
    plan = request_plan(*(layout[k] for k in ("source_table", "source_linear", "spare_table", "spare_linear")))
    cases = {}
    for row in records:
        cases.setdefault(row["case"], []).append(row)
    if set(cases) != {str(boundary - 2 + i) for i in range(4)}:
        return False
    for i, (table, linear) in enumerate(plan):
        position = boundary - 2 + i
        pages = ([layout["reserved_page"], layout["old_page"]] if position < boundary
                 else [layout["old_page"]])
        for row in cases[str(position)]:
            if (row.get("request_table") != table or row.get("linear_slot") != linear
                    or row.get("physical_pages") != pages or not row.get("neighbours_unchanged")):
                return False
    return True


class RelocationLease:
    """Temporarily repurpose allocated padding storage in a capacity-one worker.

    Move the current page into the reserved page, then use the old physical page
    for the next logical page. This forces a descending physical-page transition.
    Request rows alias the same pages only inside this serial, destructive probe.
    The ordinary scheduler allocator is never called or modified.
    """

    def __init__(self, engine, source, position, width, state_views):
        import torch

        self.engine, self.source, self.state_views = engine, source, state_views
        self.saved = []
        self.entered = False
        self.closed = False
        self.next_case = 0
        self.current = req = source.reqs[0]
        page_size = engine.config.page_size
        pool = engine.linear_state_pool
        kv = engine.kv_cache
        if (engine.config.max_running_req != 1 or len(source.reqs) != 1
                or len(source.padded_reqs) != 1 or not source.is_decode
                or not eligible_boundary(position, page_size, width, req.remain_len)
                or req.cached_len != position or req.device_len != position + 1):
            raise ValueError("relocation requires one decode request immediately before a page boundary")
        if getattr(source, "lazy_restore_pending", False):
            raise ValueError("relocation cannot borrow storage during lazy restore")
        lazy = getattr(req, "lazy_kv_restore", None)
        if lazy is not None and not lazy.complete:
            raise ValueError("relocation cannot borrow storage during lazy restore")
        source_linear = linear_slot(req)
        spare_table = engine.dummy_req.table_idx
        if (spare_table != engine.config.max_running_req or req.table_idx == spare_table
                or not 0 <= req.table_idx < engine.config.max_running_req
                or not 0 <= source_linear < pool.num_slots or pool.num_slots < 2):
            raise ValueError("relocation requires distinct allocated request and linear slots")
        spare_linear = 0 if source_linear != 0 else 1
        self.base = position // page_size * page_size
        boundary = self.base + page_size
        if boundary + page_size > engine.page_table.shape[1]:
            raise ValueError("relocation needs space for the entire next logical page")
        current = engine.page_table[req.table_idx, self.base:boundary].cpu().tolist()
        if (len(current) != page_size or current[0] < 0 or current[0] % page_size
                or current != list(range(current[0], current[0] + page_size))
                or current[0] // page_size >= engine.num_pages):
            raise ValueError("relocation requires a real contiguous current page")
        old_page = current[0] // page_size
        reserved_page = engine.num_pages
        if (kv._kv_buffer.shape[2] != reserved_page + 1
                or kv.cmp_scratch_base != (reserved_page + 1) * page_size // kv.index_ratio
                or kv._pending_ring.shape[0] <= spare_table):
            raise ValueError("relocation requires allocated KV, compressed-key and request padding storage")
        self.layout = dict(boundary=boundary, source_table=req.table_idx, source_linear=source_linear,
                           spare_table=spare_table, spare_linear=spare_linear, old_page=old_page,
                           reserved_page=reserved_page, page_size=page_size)
        self.plan = request_plan(req.table_idx, source_linear, spare_table, spare_linear)
        self.old_kv = kv._kv_buffer.select(2, old_page)
        self.reserved_kv = kv._kv_buffer.select(2, reserved_page)
        groups = page_size // kv.index_ratio
        self.old_cmp = kv._cmp_k_buffer[:, old_page * groups:(old_page + 1) * groups]
        self.reserved_cmp = kv._cmp_k_buffer[:, reserved_page * groups:(reserved_page + 1) * groups]
        self.page_row = engine.page_table[req.table_idx, self.base:boundary + page_size]
        self.spare_row = engine.page_table[spare_table]
        borrowed = copy.copy(req)
        borrowed.table_idx, borrowed.linear_slot_idx = spare_table, spare_linear
        self.borrowed_views = state_views(engine, borrowed)
        scratch = kv._cmp_k_buffer[:, kv.cmp_scratch_base + spare_table]
        self.restore_views = [self.page_row, self.spare_row, self.old_kv, self.reserved_kv,
                              self.old_cmp, self.reserved_cmp, scratch, *self.borrowed_views.values()]
        # Prepare all host-derived mapping values before any destructive copy.
        device = engine.page_table.device
        self.relocated_row = torch.cat((torch.arange(reserved_page * page_size, (reserved_page + 1) * page_size,
                                                     device=device, dtype=engine.page_table.dtype),
                                        torch.arange(old_page * page_size, (old_page + 1) * page_size,
                                                     device=device, dtype=engine.page_table.dtype)))

    def __enter__(self):
        if self.entered or self.closed:
            raise RuntimeError("relocation lease can only be entered once")
        self.engine.stream.synchronize()
        # These backups live for the whole destructive probe, outside every
        # measured forward. Keep them off the graph/checkpoint VRAM budget.
        self.saved = [(value, value.detach().to("cpu", copy=True)) for value in self.restore_views]
        self.entered = True
        try:
            self.reserved_kv.copy_(self.old_kv)
            self.reserved_cmp.copy_(self.old_cmp)
            self.old_kv.zero_()
            self.old_cmp.zero_()
            self.page_row.copy_(self.relocated_row)
            self.spare_row.copy_(self.engine.page_table[self.layout["source_table"]])
        except BaseException:
            self.close()
            raise
        return self

    def select(self, case):
        if not self.entered or self.closed or case != self.next_case or not 0 <= case < 4:
            raise RuntimeError("relocation cases must run once in order inside the lease")
        table, linear = self.plan[case]
        req = copy.copy(self.source.reqs[0])
        req.table_idx, req.linear_slot_idx = table, linear
        old, new = self.state_views(self.engine, self.current), self.state_views(self.engine, req)
        if old.keys() != new.keys():
            raise RuntimeError("relocation request state differs")
        for name, value in new.items():
            if value.data_ptr() != old[name].data_ptr():
                value.copy_(old[name])
        kv = self.engine.kv_cache
        if req.table_idx != self.current.table_idx:
            kv._cmp_k_buffer[:, kv.cmp_scratch_base + req.table_idx].copy_(
                kv._cmp_k_buffer[:, kv.cmp_scratch_base + self.current.table_idx])
        self.current = req
        self.next_case += 1
        batch = copy.copy(self.source)
        batch.reqs = batch.padded_reqs = [req]
        batch.linear_table_idx = self.source.linear_table_idx.clone().fill_(linear)
        batch.active_table_idx = self.source.active_table_idx.clone().fill_(table)
        return batch

    def neighbours(self):
        other = copy.copy(self.current)
        other.table_idx = (self.layout["source_table"] if other.table_idx == self.layout["spare_table"]
                           else self.layout["spare_table"])
        other.linear_slot_idx = (self.layout["source_linear"] if linear_slot(other) == self.layout["spare_linear"]
                                 else self.layout["spare_linear"])
        return self.state_views(self.engine, other)

    def close(self):
        if self.closed:
            return
        self.engine.stream.synchronize()
        for destination, saved in self.saved:
            destination.copy_(saved)
        self.engine.stream.synchronize()
        self.saved.clear()
        self.closed = True

    def __exit__(self, *exc):
        self.close()
