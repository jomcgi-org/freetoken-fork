"""Validation shared by KV-cache kernels and their attention callers."""


def require_unit_kv_scales(k_scale: float, v_scale: float) -> None:
    if k_scale != 1.0 or v_scale != 1.0:
        raise ValueError(
            "QSA sparse attention supports only unit FP8 KV scales; "
            f"got k_scale={k_scale}, v_scale={v_scale}"
        )
