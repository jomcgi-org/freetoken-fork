from __future__ import annotations

import pytest
import torch


class _KernelCapture:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


@pytest.mark.parametrize("rope_dim", [0, 16])
@pytest.mark.parametrize("force_splits", [0, 2])
def test_sparse_launch_shapes_and_constexprs(monkeypatch, rope_dim, force_splits):
    """Build both launch variants on CPU and inspect their tensor and constexpr args."""
    from freetoken.kernel.triton import glm_dsa_sparse as sparse

    stage = _KernelCapture()
    merge = _KernelCapture()
    if force_splits:
        monkeypatch.setattr(sparse, "_glm_dsa_splitk_kernel", stage)
        monkeypatch.setattr(sparse, "_glm_dsa_merge_kernel", merge)
    else:
        monkeypatch.setattr(sparse, "_glm_dsa_sparse_kernel", stage)

    b, m, h, value_dim, rows, topk = 1, 1 if force_splits else 3, 2, 16, 7, 4
    latent_dim = value_dim + rope_dim
    q = torch.randn(b, m, h, latent_dim)
    pool = torch.randn(rows, latent_dim)
    idx = torch.tensor([[[0, 2, 4, 6]]], dtype=torch.int32).expand(b, m, topk)
    counts = torch.full((b, m), topk, dtype=torch.int32)

    out = sparse.glm_dsa_sparse_attn(
        q,
        pool,
        idx,
        latent_dim**-0.5,
        counts=counts,
        d_v=value_dim,
        force_splits=force_splits,
    )

    assert out.shape == (b, m, h, value_dim)
    assert len(stage.launches) == 1
    grid, args, constexprs = stage.launches[0]
    assert grid == ((m * force_splits, b, 1) if force_splits else (m, b, 1))
    expected = {"D_V", "D_R", "BLOCK_H", "BLOCK_T", "HAS_COUNTS"}
    if force_splits:
        expected.add("NUM_SPLITS")
    assert set(constexprs) - {"num_warps", "num_stages"} == expected
    assert constexprs["D_V"] == value_dim
    assert constexprs["D_R"] == rope_dim
    assert constexprs["HAS_COUNTS"] is True

    assert args[0].shape == (b, m, h, latent_dim)
    assert args[1].shape == (rows, latent_dim)
    if force_splits:
        assert args[2].shape == (b, m, h, force_splits, value_dim)
        assert args[3].shape == (b, m, h, force_splits)
        assert args[4].shape == (b, m, topk)
        assert args[5].shape == (b, m)
        assert len(merge.launches) == 1
        merge_grid, merge_args, merge_constexprs = merge.launches[0]
        assert merge_grid == (m, b, h)
        assert set(merge_constexprs) - {"num_warps"} == {"D_V", "NUM_SPLITS"}
        assert merge_args[0].shape == (b, m, h, force_splits, value_dim)
        assert merge_args[1].shape == (b, m, h, force_splits)
        assert merge_args[2].shape == (b, m, h, value_dim)
    else:
        assert args[2].shape == (b, m, h, value_dim)
        assert args[3].shape == (b, m, topk)
        assert args[4].shape == (b, m)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("force_splits,queries", [(0, 3), (2, 1)])
def test_zero_rope_sparse_attention_matches_torch(force_splits, queries):
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

    torch.manual_seed(11)
    heads, value_dim, rows, live = 2, 16, 8, 5
    q = torch.randn(1, queries, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(rows, value_dim, device="cuda", dtype=torch.bfloat16)
    selected = torch.tensor([6, 1, 4, 0, 7], device="cuda", dtype=torch.int32)
    idx = torch.full((1, queries, 7), -1, device="cuda", dtype=torch.int32)
    idx[0, :, :live] = selected
    counts = torch.full((1, queries), live, device="cuda", dtype=torch.int32)
    scale = value_dim**-0.5

    out = glm_dsa_sparse_attn(
        q,
        pool,
        idx,
        scale,
        counts=counts,
        d_v=value_dim,
        force_splits=force_splits,
    )

    keys = pool[selected.long()].float()
    scores = q[0].float() @ keys.T * scale
    ref = scores.softmax(dim=-1) @ keys
    assert (out[0].float() - ref).abs().max().item() < 2e-2
