"""Distance bucketizer.

This is the *only* component that changes when real metric depth lands.
Phrase templates and the composer stay untouched (Requirement 15). The
mapping is a single function `bucketize(depth_m, cfg)` — replace its body,
and the entire system speaks truthful numbers in the same phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

DistanceBucket = Literal["immediate", "near", "mid", "far"]
_BUCKET_ORDER: tuple[DistanceBucket, ...] = ("immediate", "near", "mid", "far")

_DEFAULT_PHRASES: dict[DistanceBucket, str] = {
    "immediate": "right in front of you",
    "near": "about 6 feet ahead",
    "mid": "about 10 feet ahead",
    "far": "about 30 feet ahead",
}


@dataclass(frozen=True)
class DistanceConfig:
    """Thresholds and phrases that drive `bucketize`. Loaded from YAML."""

    immediate_max_m: float = 1.2
    near_max_m: float = 2.2
    mid_max_m: float = 3.2
    default_bucket: DistanceBucket = "near"
    phrases: dict[DistanceBucket, str] = field(
        default_factory=lambda: dict(_DEFAULT_PHRASES)
    )


def load_distance_config(yaml_dict: dict[str, Any] | None) -> DistanceConfig:
    """Build a `DistanceConfig` from the ``distance:`` block of default.yaml.

    Missing fields fall back to the design defaults. Unknown bucket names in
    the phrases map are ignored. Always returns a fully-populated phrases
    dict so `bucketize` never trips on a `KeyError`.
    """
    block = (yaml_dict or {}).get("distance", {}) if isinstance(yaml_dict, dict) else {}

    phrases = dict(_DEFAULT_PHRASES)
    raw_phrases = block.get("phrases") or {}
    if isinstance(raw_phrases, dict):
        for bucket, phrase in raw_phrases.items():
            if bucket in _BUCKET_ORDER and isinstance(phrase, str) and phrase:
                phrases[bucket] = phrase  # type: ignore[index]

    default_bucket = block.get("default_bucket", "near")
    if default_bucket not in _BUCKET_ORDER:
        default_bucket = "near"

    return DistanceConfig(
        immediate_max_m=float(block.get("immediate_max_m", 1.2)),
        near_max_m=float(block.get("near_max_m", 2.2)),
        mid_max_m=float(block.get("mid_max_m", 3.2)),
        default_bucket=default_bucket,  # type: ignore[arg-type]
        phrases=phrases,
    )


def bucketize(
    depth_m: Optional[float],
    cfg: DistanceConfig | None = None,
) -> tuple[DistanceBucket, str]:
    """Map a depth-in-meters reading to a bucket and a spoken phrase.

    `depth_m=None` returns the configured default bucket. Phrases are taken
    verbatim from the config — the function never speaks a raw float. This
    is the single substitution point when real metric depth lands; the
    phrase strings themselves are unchanged.
    """
    cfg = cfg or DistanceConfig()
    if depth_m is None:
        return cfg.default_bucket, cfg.phrases[cfg.default_bucket]
    if depth_m <= cfg.immediate_max_m:
        return "immediate", cfg.phrases["immediate"]
    if depth_m <= cfg.near_max_m:
        return "near", cfg.phrases["near"]
    if depth_m <= cfg.mid_max_m:
        return "mid", cfg.phrases["mid"]
    return "far", cfg.phrases["far"]


__all__ = [
    "DistanceBucket",
    "DistanceConfig",
    "bucketize",
    "load_distance_config",
]
