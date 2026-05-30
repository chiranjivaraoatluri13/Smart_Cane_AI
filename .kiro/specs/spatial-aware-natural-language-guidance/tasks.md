# Implementation Plan

This plan turns the design into incremental, test-first tasks. Each task references the requirements it implements. PBT-tagged tasks (marked `(PBT)`) implement one of the 11 correctness properties from `design.md` and must use `hypothesis` with at least 100 examples and the comment header `# Feature: spatial-aware-natural-language-guidance, Property N: <text>`.

The order is engineered so the existing 47 tests stay green at every step:

1. Foundations and data models first (no behavior change).
2. New perception helpers (purely additive).
3. New reasoning components (pluggable; legacy path stays callable).
4. New output components (composer + voice queue).
5. Wire into runner and phone server (additive; `--legacy-reasoner` flag preserves CLI behavior).
6. Phone client + `/set_destination`.
7. Performance benchmark and final acceptance.

---

- [x] 1. Configuration and settings scaffolding
  - [x] 1.1 Add new `Settings` fields (`min_lane_walkable_ratio`, `stairs_detector_enabled`, `stairs_min_edge_density`, `stairs_min_confidence`, `status_update_interval_sec`, `voice_cooldowns`, `composer_seed`, `benchmark_mode`, `phrases_path`) in `navigation/config.py` with the defaults from the design.
    - _Requirements: 2.5, 5.2, 5.4, 9.1, 9.4, 9.6, 9.8, 10.1, 11.5, 14.3_
  - [x] 1.2 Add `distance:`, `voice:`, `spatial:` blocks to `config/default.yaml` and a small `load_distance_config(yaml_dict) -> DistanceConfig` helper in `navigation/output/distance.py`.
    - _Requirements: 6.5, 9.1, 15.2_
  - [x] 1.3 Create `config/phrases.yaml` with the 15 scenario tags and 3–5 paraphrases each, exactly as listed in the design's Template_Library section.
    - _Requirements: 10.1, 10.2, 10.7_

- [x] 2. Data model extensions
  - [x] 2.1 Add `per_side_class_pixels: dict[Side, dict[str, float]] | None = None` and `per_side_walkable_ratio: dict[Side, float] | None = None` to `SegmentationResult` in `navigation/models.py`. Default `None`. Add `Side = Literal["left","center","right"]` re-exported from `navigation/models.py`.
    - _Requirements: 1.2, 1.3, 2.2, 2.3, 12.6_
  - [x] 2.2 Create `navigation/reasoning/facts.py` with `HazardEntry`, `RouteCue`, `StairsResult`, `DistanceBucket`, `ApproachDirection`, and `GuidanceFacts` dataclasses (or pydantic models — match existing style). Include `GuidanceFacts.summary_dict()` for HUD/JSON.
    - _Requirements: 3.1, 3.5, 4.8, 5.3, 7.1_

- [x] 3. Per-side perception helpers
  - [x] 3.1 Create `navigation/perception/spatial.py` with `_side_slices(width)`, `_per_side_class_pixels(class_map, id_to_name, weight_map)`, `_per_side_walkable_ratio(class_map, id_to_name, walkable_classes)`. Reuse the existing `_region_weight_map` from `segmentation.py`.
    - _Requirements: 1.1, 1.5, 2.1_
  - [x] 3.2 Wire the helpers into `YoloSegmenter._parse_semantic` so per-side fields are populated in the same `for cls_id in np.unique(class_map):` pass. Keep `_parse_instance` and `_mock` returning `None`/zeroed dicts (Req 1.6).
    - _Requirements: 1.1, 1.2, 1.3, 1.6, 12.6_
  - [x] 3.3 (PBT) Write `tests/test_spatial.py::test_per_side_counts_round_trip_to_global` — Property 1 — generate random `class_map`s, verify `sum(per_side[s][c]) == global[c]` within tolerance of 1, and that `width(left) == width(right)` with remainder columns in `center`. 100 hypothesis examples.
    - _Requirements: 1.1, 1.4, 1.5_
  - [x] 3.4 (PBT) Write `tests/test_spatial.py::test_per_side_walkable_ratio_in_unit_interval` — Property 2 — generate random class maps and walkable sets, verify each side's ratio in `[0, 1]`. 100 hypothesis examples.
    - _Requirements: 2.1, 2.2_
  - [x] 3.5 Write `tests/test_spatial.py::test_mock_returns_zeroed_per_side_dicts` (example) and `test_per_side_counts_split_evenly` (example) for non-property assertions.
    - _Requirements: 1.5, 1.6_

- [ ] 4. Distance bucketizer
  - [x] 4.1 Create `navigation/output/distance.py` with `DistanceBucket`, `DistanceConfig`, and `bucketize(depth_m, cfg) -> tuple[DistanceBucket, str]` exactly as in the design.
    - _Requirements: 6.1, 6.2, 6.5, 15.1, 15.4_
  - [x] 4.2 (PBT) Write `tests/test_distance.py::test_bucketize_threshold_boundaries` and `::test_phrase_has_hedge_word_and_int_feet` and `::test_phrase_has_no_unsubstituted_placeholders` and `::test_none_depth_returns_default_bucket` — Property 7 — generate random `depth_m` floats and `None`, assert monotone bucket assignment, hedge word + integer feet for non-immediate, no decimal-style numbers, default bucket on `None`. 100 hypothesis examples.
    - _Requirements: 6.1, 6.3, 6.4, 15.4_
  - [x] 4.3 Write `tests/test_distance.py::test_immediate_bucket_no_number` (example) — bucketize(0.5, cfg) returns `("immediate", "right in front of you")` exactly.
    - _Requirements: 6.2_

- [ ] 5. Stairs detector
  - [x] 5.1 Create `navigation/perception/stairs.py` with `StairsDetector` class. Use `cv2.Sobel(masked, CV_32F, 0, 1, ksize=3)` over the bottom-30% walkable region, compute per-row density, threshold against `stairs_min_edge_density`, require row span >= 30% walkable width. Return `StairsResult` dataclass.
    - _Requirements: 5.1, 5.2, 5.6, 5.7, 14.1, 14.3_
  - [x] 5.2 (PBT) Write `tests/test_stairs.py::test_flag_on_synthetic_step` — Property 11 — generate synthetic frames with controlled horizontal-gradient strengths in the bottom 30% of a walkable mask, assert flag matches the threshold and confidence in `[0, 1]`. 100 hypothesis examples.
    - _Requirements: 5.2_
  - [x] 5.3 Write `tests/test_stairs.py::test_no_flag_on_smooth_floor` (example) and `::test_no_walkable_returns_no_flag_no_exception` (edge case).
    - _Requirements: 5.6_
  - [x] 5.4 Write `tests/test_stairs.py::test_returns_stairs_result_dataclass` and `::test_disabled_returns_no_flag_no_phrase` (contract examples).
    - _Requirements: 14.1, 14.3_

- [ ] 6. Trend tracker
  - [x] 6.1 Create `navigation/reasoning/trend.py` with `TrendTracker` per the design (per-category 6-slot ring buffer, `update()`, `classify(category)`, `classify_all()`).
    - _Requirements: 4.1, 4.2, 4.8, 4.9_
  - [x] 6.2 Refactor `AlertTracker._weighted_counts_per_category` into a free function `weighted_counts_per_category(seg)` shared with `TrendTracker`. Keep `AlertTracker.update()` signature and behavior unchanged so `tests/test_alerts.py` passes.
    - _Requirements: 12.5_
  - [x] 6.3 (PBT) Write `tests/test_trend.py::test_classify_crossing_left_to_right` and `::test_classify_crossing_right_to_left` — Property 5 — generate per-side weight sequences whose centroid increases (or decreases) monotonically by >= 0.15, assert the matching crossing label. Use `hypothesis.strategies.lists(tuples(...), min_size=3, max_size=6)`. 100 examples each.
    - _Requirements: 4.3, 4.4_
  - [x] 6.4 (PBT) Write `tests/test_trend.py::test_classify_closing_in_when_growing`, `::test_classify_receding_when_shrinking`, `::test_classify_static_when_no_movement`, `::test_classify_returns_valid_label`, `::test_buffer_size_bounded` — Property 6 — assert label stays in the five-string set, buffer never exceeds size 6, and that growth/recede thresholds produce the right labels. 100 examples each.
    - _Requirements: 4.1, 4.2, 4.5, 4.6, 4.7_

- [x] 7. SpatialReasoner
  - [x] 7.1 Create `navigation/reasoning/spatial_reasoner.py` with `SpatialReasoner.decide()` exactly per the design's flow (vision_stop short-circuit → hazards_by_side → walkable check → route_cue blend → CARE-direction fallback → build `GuidanceFacts`). Vision STOP must be computed first and short-circuit every other branch (Req 13).
    - _Requirements: 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 4.8, 7.1, 7.5, 13.1_
  - [x] 7.2 Add `_next_route_cue(interpreter, position) -> RouteCue | None` helper inside `spatial_reasoner.py` that pulls from the existing `MapGuidance.decide()` (or from a new `MapGuidance.next_cue(position)` if cleaner) without rewriting map logic.
    - _Requirements: 7.1, 7.4_
  - [x] 7.3 (PBT) Write `tests/test_spatial_reasoner.py::test_vision_stop_overrides_all` — Property 4 — generate combinations of `(care.hazard_detected, center_obstacle_ratio, walkable_by_side, route_cue)`, assert when vision_stop condition holds, decision is STOP and `facts.route_cue is None`. 100 examples.
    - _Requirements: 3.4, 7.3, 7.5, 13.1, 13.3, 13.4_
  - [x] 7.4 (PBT) Write `tests/test_spatial_reasoner.py::test_directional_move_only_when_target_walkable_at_least_center` — Property 3 — generate non-stop facts; whenever decision is MOVE_LEFT, assert `walkable_by_side["left"] >= walkable_by_side["center"]` (mirror for MOVE_RIGHT). 100 examples.
    - _Requirements: 2.4_
  - [x] 7.5 Write example tests `::test_combines_left_and_right_hazards`, `::test_route_cue_merged_into_facts`, `::test_map_and_vision_blend_when_clear`, `::test_slow_down_when_no_lane_walkable`, `::test_vision_stop_drops_route`.
    - _Requirements: 2.5, 3.1, 3.2, 7.1, 7.6_

- [ ] 8. PhraseComposer
  - [x] 8.1 Create `navigation/reasoning/composer.py` with `PhraseComposer` per the design: load `phrases.yaml` once, scenario routing logic in `_scenario_for(facts)`, placeholder substitution, no-repeat tracking, missing-template fallback.
    - _Requirements: 5.4, 5.5, 6.6, 7.2, 7.3, 7.4, 7.6, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_
  - [x] 8.2 Implement `bucketize` integration: composer uses `facts.distance_phrase` (already populated by `SpatialReasoner` from `bucketize`) so the composer never inlines distance thresholds (Req 15.1).
    - _Requirements: 6.6, 15.1_
  - [x] 8.3 (PBT) Write `tests/test_composer.py::test_paraphrase_no_repeat_in_window` and `::test_no_unsubstituted_placeholders` — Property 8 — generate random `GuidanceFacts`, verify rendered phrase has no `{...}` substrings and that two consecutive same-tag calls produce different paraphrases when ≥2 are configured. 100 examples.
    - _Requirements: 10.3, 10.6, 10.8_
  - [x] 8.4 Write example tests `::test_loads_phrases_yaml`, `::test_missing_template_falls_back`, `::test_under_populated_tag_warns_once`, `::test_stairs_warning_low_conf_phrase`, `::test_stairs_with_stop_prepended`, `::test_route_blend_phrase_contains_both_pieces`, `::test_no_route_cue_renders_vision_only_phrase`, `::test_composer_calls_bucketize_via_facts_phrase`.
    - _Requirements: 5.4, 5.5, 7.2, 7.4, 10.1, 10.4, 10.5, 15.1_

- [ ] 9. VoiceQueue
  - [ ] 9.1 Create `navigation/output/voice_queue.py` with `VoiceTier` literal, `TIER_PRIORITY`, `VoiceItem` dataclass, `VoiceQueue` (`enqueue`, `tick`, internal `_last_spoken_at` per tier). Inject `clock` for testability.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 13.2_
  - [ ] 9.2 (PBT) Write `tests/test_voice_queue.py::test_higher_tier_preempts_lower_tier` — Property 9 — generate random enqueue sequences, assert `tick()` returns the highest-priority item that hasn't been cooldown-gated. 100 examples.
    - _Requirements: 9.2, 9.3, 9.5, 9.7, 13.2_
  - [ ] 9.3 (PBT) Write `tests/test_voice_queue.py::test_map_turn_cooldown`, `::test_status_update_interval`, `::test_per_tier_cooldowns_independent` — Property 10 — drive `VoiceQueue` with a fake clock; verify per-tier cooldowns hold and are independent. 100 examples each.
    - _Requirements: 9.4, 9.6, 9.8_
  - [ ] 9.4 Write example tests `::test_stop_preempts`, `::test_tts_disabled_logs_without_error`.
    - _Requirements: 9.2, 9.9_

- [x] 10. Pipeline runner integration
  - [x] 10.1 Update `navigation/pipeline/runner.py::process_frame` to accept and use the new components: `spatial_reasoner`, `composer`, `voice_queue`, `trend_tracker`, `stairs_detector`, plus `use_legacy_reasoner` flag. Default the new path on; legacy path runs only when `use_legacy_reasoner=True`. Existing `interpreter` parameter is preserved for the legacy branch. The `record` dict gains an optional `"facts"` key; all existing keys keep their shape.
    - _Requirements: 3.1, 7.1, 9.1, 11.5, 12.5, 13.4_
  - [x] 10.2 Update `run_live` and `run_image` to construct and inject the new components alongside the existing ones.
    - _Requirements: 12.5_
  - [x] 10.3 Add `Settings.benchmark_mode` instrumentation: when set, `process_frame` populates `record["timings_ms"] = {seg, depth, care, stairs, trend, reasoner, composer, voice}`.
    - _Requirements: 11.5_
  - [x] 10.4 Add `--legacy-reasoner` and `--benchmark` flags to `navigation/cli.py`. Default both to off. `--legacy-reasoner` flips `use_legacy_reasoner=True`; `--benchmark` flips `Settings.benchmark_mode=True`.
    - _Requirements: 11.5, 12.5_
  - [x] 10.5 Extend `tests/test_runner.py` with `::test_benchmark_mode_emits_per_stage_timing` (example) and `::test_pipeline_smoke_through_composer_and_queue` (integration). Mark performance test with `pytest.mark.benchmark`. Do NOT modify any existing assertion.
    - _Requirements: 11.1, 11.5, 12.5_

- [x] 11. Phone server `/set_destination`
  - [x] 11.1 In `phone_server.py`, instantiate `SpatialReasoner`, `PhraseComposer`, `VoiceQueue`, `TrendTracker`, `StairsDetector` at module load alongside the existing components.
    - _Requirements: 12.4_
  - [x] 11.2 Replace the inline `/process_frame` body with a call to `navigation.pipeline.runner.process_frame()` injecting all components, passing `position` from form fields. Keep the `_pipeline_lock` wrapping.
    - _Requirements: 7.1, 12.4_
  - [x] 11.3 Add the `/set_destination` Flask route. Validate `address`; geocode via `geocode_address`; on success, mutate the running `interpreter.settings` via `model_copy({dest_lat, dest_lon, use_map_guidance: True})` and reset `_map_guidance` so the next frame re-fetches the route. Reuse `_pipeline_lock`.
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9_
  - [x] 11.4 Write `tests/test_phone_server.py` (Flask test client). Cover: `test_set_destination_geocodes` (mocked), `test_set_destination_400_on_missing`, `test_set_destination_422_on_no_geocode`, `test_set_destination_without_gps_defers_route_fetch`, `test_set_destination_posts_address_form_field`.
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 8.9_

- [x] 12. Phone client UI
  - [x] 12.1 Add the `#dest-bar` div to `phone_client.html` containing the `Destination` text input and `Set` button. Position it at the top of the screen so it does not visually overlap Start/Stop.
    - _Requirements: 8.1_
  - [x] 12.2 Add the JS `setDestination()` handler that POSTs `application/x-www-form-urlencoded` with the `address` field to `/set_destination` and writes the response to the existing `dbg()` debug pane. Hook both the button click and Enter key in the input.
    - _Requirements: 8.2, 8.4, 8.5, 8.6_

- [x] 13. Final acceptance and benchmarking
  - [x] 13.1 Run the full test suite (`pytest`). All 47 existing tests must still pass. New tests must all pass. Document any expected hypothesis examples in flake-resistant form (use `@settings(deadline=None)` on slow property tests).
    - _Requirements: 12.5_
  - [x] 13.2 Run the benchmark integration test on the reference machine and record the median frame time. Target: < 100 ms median over 100 frames in fast profile. If above, identify the offender via the per-stage timing record from task 10.3.
    - _Requirements: 11.1, 11.5_
  - [x] 13.3 Sanity-check that no new heavy dependency was added to `pyproject.toml` (only `hypothesis` may have been added as a dev dep — confirm). Confirm no upward imports introduced (manual review or simple grep).
    - _Requirements: 12.1, 12.2, 12.3, 12.4_
