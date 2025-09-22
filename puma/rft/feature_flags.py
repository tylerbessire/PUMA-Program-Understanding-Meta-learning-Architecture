"""Feature flags and deterministic limit helpers for RFT."""
# [S:OPER v1] feature_flag=RFT pass

from __future__ import annotations

import os
from typing import Dict, Optional

from .types import Limits

RFT_ENABLED = os.environ.get("PUMA_RFT", "0").lower() in {"1", "true", "yes"}
RFT_DEFAULT_LIMITS: Dict[str, float | int] = {
    "PLIANCE_STEPS": 100,
    "TRACKING_BUDGET": 100,
    "THRESH": 0.5,
    "OUTER_BUDGET": 10,
}


def _seed_from_env() -> int:
    seed_value = os.environ.get("PUMA_SEED")
    if seed_value is None:
        return 0
    try:
        return int(seed_value)
    except ValueError:
        return abs(hash(seed_value)) % (2**32)


def build_limits(overrides: Optional[Dict[str, float | int]] = None) -> Limits:
    """Return :class:`Limits` merged with optional overrides."""

    params = dict(RFT_DEFAULT_LIMITS)
    if overrides:
        params.update(overrides)
    limits = Limits(
        pliance_steps=int(params["PLIANCE_STEPS"]),
        tracking_budget=int(params["TRACKING_BUDGET"]),
        thresh=float(params["THRESH"]),
        outer_budget=int(params["OUTER_BUDGET"]),
    )
    # Deterministic seed for downstream contexts.
    os.environ.setdefault("PUMA_SEED", str(_seed_from_env()))
    return limits


__all__ = ["RFT_ENABLED", "RFT_DEFAULT_LIMITS", "build_limits"]
