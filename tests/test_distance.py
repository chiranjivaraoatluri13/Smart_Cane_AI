"""Tests for navigation.output.distance — bucketizer and phrases."""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings, strategies as st

from navigation.output.distance import (
    DistanceConfig,
    _BUCKET_ORDER,
    bucketize,
    load_distance_config,
)

_ALLOWED_BUCKETS = {"immediate", "near", "mid", "far"}
_HEDGE_WORDS = ("about", "roughly", "around")
_BUCKET_RANK = {b: i for i, b in enumerate(_BUCKET_ORDER)}


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


# Feature: spatial-aware-natural-language-guidance, Property 7: Distance bucket monotone in depth
@given(depth=st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)))
@settings(max_examples=100, deadline=None)
def test_bucketize_returns_known_bucket_and_phrase(depth):
    cfg = DistanceConfig()
    bucket, phrase = bucketize(depth, cfg)
    assert bucket in _ALLOWED_BUCKETS
    assert isinstance(phrase, str) and phrase
    # No raw decimals like 32.4 in any phrase.
    assert re.search(r"\d+\.\d+", phrase) is None


# Feature: spatial-aware-natural-language-guidance, Property 7: Distance bucket monotone in depth
@given(
    a=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_bucketize_is_monotone_in_depth(a, b):
    """Increasing depth never moves the bucket toward a smaller bucket."""
    cfg = DistanceConfig()
    lo, hi = (a, b) if a <= b else (b, a)
    bucket_lo, _ = bucketize(lo, cfg)
    bucket_hi, _ = bucketize(hi, cfg)
    assert _BUCKET_RANK[bucket_hi] >= _BUCKET_RANK[bucket_lo]


# Feature: spatial-aware-natural-language-guidance, Property 7: hedge word + integer feet for non-immediate
def test_phrase_has_hedge_word_and_int_feet_for_non_immediate():
    cfg = DistanceConfig()
    for bucket in ("near", "mid", "far"):
        phrase = cfg.phrases[bucket]
        assert any(h in phrase for h in _HEDGE_WORDS), (
            f"phrase {phrase!r} missing hedge word"
        )
        m = re.search(r"\b(\d+)\s+feet\b", phrase)
        assert m is not None, f"phrase {phrase!r} missing integer-feet token"
        # No decimal-style numbers.
        assert re.search(r"\d+\.\d+", phrase) is None


def test_phrase_has_no_unsubstituted_placeholders():
    cfg = DistanceConfig()
    for phrase in cfg.phrases.values():
        assert "{" not in phrase and "}" not in phrase


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_immediate_bucket_no_number():
    """Requirement 6.2 — immediate uses 'right in front of you', no feet."""
    cfg = DistanceConfig()
    bucket, phrase = bucketize(0.5, cfg)
    assert bucket == "immediate"
    assert phrase == "right in front of you"
    assert "feet" not in phrase


def test_bucketize_threshold_boundaries():
    cfg = DistanceConfig()
    assert bucketize(0.0, cfg)[0] == "immediate"
    assert bucketize(1.2, cfg)[0] == "immediate"
    assert bucketize(1.21, cfg)[0] == "near"
    assert bucketize(2.2, cfg)[0] == "near"
    assert bucketize(2.21, cfg)[0] == "mid"
    assert bucketize(3.2, cfg)[0] == "mid"
    assert bucketize(3.21, cfg)[0] == "far"
    assert bucketize(99.0, cfg)[0] == "far"


def test_none_depth_returns_default_bucket():
    cfg = DistanceConfig()
    bucket, phrase = bucketize(None, cfg)
    assert bucket == cfg.default_bucket
    assert phrase == cfg.phrases[cfg.default_bucket]


def test_load_distance_config_from_yaml_block():
    yaml_dict = {
        "distance": {
            "immediate_max_m": 1.0,
            "near_max_m": 2.0,
            "mid_max_m": 3.0,
            "default_bucket": "mid",
            "phrases": {
                "immediate": "in your face",
                "near": "about 5 feet away",
                "mid": "about 9 feet away",
                "far": "about 28 feet away",
                "ignored_bucket": "should be dropped",
            },
        }
    }
    cfg = load_distance_config(yaml_dict)
    assert cfg.immediate_max_m == 1.0
    assert cfg.near_max_m == 2.0
    assert cfg.default_bucket == "mid"
    assert cfg.phrases["immediate"] == "in your face"
    assert "ignored_bucket" not in cfg.phrases


def test_load_distance_config_missing_block_returns_defaults():
    cfg = load_distance_config(None)
    assert cfg.default_bucket == "near"
    assert cfg.phrases["immediate"] == "right in front of you"


def test_load_distance_config_invalid_default_bucket_falls_back_to_near():
    cfg = load_distance_config({"distance": {"default_bucket": "nonsense"}})
    assert cfg.default_bucket == "near"


def test_bucketize_uses_default_config_when_none_given():
    bucket, phrase = bucketize(0.5)
    assert bucket == "immediate"
    assert phrase == "right in front of you"
