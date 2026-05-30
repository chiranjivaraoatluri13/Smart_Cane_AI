# Requirements Document

## Introduction

This feature upgrades the existing assistive-navigation system from short, rule-based commands ("stop", "go forward") to a spatially aware, natural-sounding direction guidance system. It is "Path A" — smart rule-based composition with a template phrase library. No LLM, VLM, or extra ML model is introduced. The system continues to derive its safety signals from the current YOLO Cityscapes segmenter, the CARE reasoning layer, the 3-bucket segmentation-based depth proxy, and OSRM walking routes from the maps layer.

The new behavior:

- The pipeline computes per-side (left/center/right third of the frame) class statistics and walkable-space ratios so reasoning can talk about "your left" vs "ahead" vs "your right".
- A multi-object reasoner combines all hazards in a single decision instead of picking one.
- A short ring buffer tracks per-category centroid trends across recent frames and labels approach direction (e.g. "crossing left to right" vs "static in front").
- A heuristic stairs/curb detector flags horizontal-edge / luminance discontinuities in the bottom 30% of the walkable region.
- Distances are spoken in feet using hedge words ("about 30 feet ahead", "right in front of you"), backed by the existing 3-bucket proxy. When real metric depth lands later, only the bucketizer is replaced; phrase templates are stable.
- Map and vision are blended into one coherent phrase per turn cue. Vision-first on conflict: a vision STOP always overrides any route guidance.
- The phone client gets an address textbox and the server gains a `/set_destination` endpoint that geocodes via the existing Nominatim integration.
- Voice output gets a priority queue with per-tier cooldowns and a quiet conversational status update every ~10 s when nothing else is talking.
- All spoken output is generated from a small data-driven template library with 3–5 paraphrases per scenario; phrases are not hard-coded in the reasoning layer.

The work is constrained to keep the pipeline at ~100 ms per inference frame on a CPU-only Intel Core Ultra 7 256V, to keep the new spatial logic O(1) per frame, to introduce no new heavy dependencies, and to keep all 47 existing tests green. Architectural rules from `ARCHITECTURE_FIX.md` are preserved: the new phrase composer lives in the reasoning layer; perception and utils never import from higher layers.

## Glossary

- **System**: The entire assistive-navigation application running on the laptop with the phone-as-camera client.
- **Pipeline**: The orchestration loop in `navigation/pipeline/runner.py` and the equivalent path in `phone_server.py`.
- **Segmenter**: `navigation/perception/segmentation.py` (YoloSegmenter), produces a `SegmentationResult`.
- **Reasoner**: The multi-object reasoning component in the reasoning layer that consumes a `PerceptionBundle` and emits a structured `GuidanceFacts` object describing what should be said.
- **Composer**: The phrase composer in the reasoning layer that turns `GuidanceFacts` into a final spoken phrase using the Template_Library. Lives in `navigation/reasoning/composer.py` per the architecture rule that composer lives in reasoning.
- **Template_Library**: The data-driven set of paraphrase templates loaded from configuration (e.g. `config/phrases.yaml`) by the Composer.
- **Stairs_Detector**: The heuristic stairs/curb detector that scans the bottom 30% of the walkable region for horizontal-edge / luminance discontinuities.
- **Trend_Tracker**: A small in-memory ring buffer (per category) that records per-category centroids across the last N frames and labels approach direction.
- **Voice_Queue**: The priority-aware speech scheduler in the output layer that owns per-tier cooldowns and decides what is spoken next.
- **Map_Guidance**: The existing `navigation/maps/guidance.py` route-following component that yields turn-by-turn cues from an OSRM RoutePlan.
- **Phone_Client**: The browser-based client in `phone_client.html` that captures camera, GPS, and compass and forwards them to the server.
- **Phone_Server**: The Flask server in `phone_server.py` that exposes `/process_frame`, `/health`, `/stats`, and (new) `/set_destination`.
- **SegmentationResult**: The pydantic model in `navigation/models.py` returned by the Segmenter. Carries existing fields (class_map, obstacle_pixels, obstacle_pixels_weighted, walkable_ratio, metadata) plus the new optional per-side fields introduced by this feature.
- **Side**: One of `left`, `center`, `right`. Defined as the left, center, and right third of the frame width respectively.
- **Lane**: The walkable region within a single Side; "free-space lane" means the walkable-pixel ratio inside that Side.
- **GuidanceFacts**: A new structured object emitted by the Reasoner and consumed by the Composer. Contains per-side class summaries, per-side walkable ratios, approach-direction labels per category, stairs/curb flag, distance bucket, and any active route cue from Map_Guidance.
- **Distance_Bucket**: One of `immediate` (~1 m / "right in front of you"), `near` (~2 m / "about 6 feet ahead"), `mid` (~3 m / "about 10 feet ahead"), or `far` (≥ ~3 m / "about 30 feet ahead"). The bucket-to-feet mapping lives in configuration and is the only part replaced when real metric depth arrives.
- **Voice_Tier**: One of `vision_stop`, `directional_warning`, `map_turn`, `approach_alert`, `status_update`. Higher tiers preempt or interrupt lower tiers per the priority rules.
- **Status_Update**: A quiet conversational scene description spoken at most once every ~10 s when no higher-tier voice item is queued.
- **Cooldown**: A per-tier minimum interval between two consecutive utterances of that tier.
- **Vision_STOP**: A `NavigationCommand.STOP` decision originating from the vision/CARE path (i.e. `hazard_detected` is true or the heuristic obstacle gate fires).
- **Architecture_Rules**: The layering rules from `ARCHITECTURE_FIX.md` — utils → config → models → capture → perception → reasoning → maps → output → pipeline → cli, with no upward imports.

## Requirements

### Requirement 1: Per-side spatial detection

**User Story:** As a blind walker, I want the system to know what is on my left, ahead, and on my right, so that the spoken guidance can describe scene contents directionally instead of as a single global blob.

#### Acceptance Criteria

1. WHEN the Segmenter produces a class map for a frame, THE Segmenter SHALL split the class map into three vertical Sides (left third, center third, right third by frame width) and compute, for each Side, a per-class weighted pixel count using the existing region-weight map.
2. WHEN the Segmenter produces a class map for a frame, THE Segmenter SHALL populate a new optional `per_side_class_pixels` field on the SegmentationResult with shape `{left: {class_name: weighted_count, ...}, center: {...}, right: {...}}`.
3. WHERE a SegmentationResult is constructed without per-side fields, THE Segmenter SHALL leave `per_side_class_pixels` as `None` and existing callers SHALL continue to function without modification.
4. WHEN per-side counts are computed, THE Segmenter SHALL ensure that, for every class, the sum of the three Side counts equals the existing global `obstacle_pixels_weighted` contribution for that class within a tolerance of 1 weighted pixel.
5. WHEN the frame width is not evenly divisible by three, THE Segmenter SHALL assign any remainder columns to the center Side so that left and right Sides have equal width.
6. WHEN the Segmenter is in mock mode (`dry_run=True` or no model loaded), THE Segmenter SHALL still produce a `per_side_class_pixels` value populated with zeroed dictionaries so downstream code receives a stable shape.

### Requirement 2: Per-side free-space awareness

**User Story:** As a blind walker, I want the system to know which Side has open walkable space, so that "move left" or "move right" only fires when that Side is actually clear.

#### Acceptance Criteria

1. WHEN the Segmenter produces a class map for a frame, THE Segmenter SHALL compute a walkable-pixel ratio for each Side using only that Side's columns and the existing walkable-class set from `config/default.yaml`.
2. WHEN per-side walkable ratios are computed, THE Segmenter SHALL populate a new optional `per_side_walkable_ratio` field on the SegmentationResult with shape `{left: float, center: float, right: float}`, each in the closed interval `[0.0, 1.0]`.
3. WHERE per-side walkable ratios are not computed (legacy code path or unsupported model output), THE Segmenter SHALL leave `per_side_walkable_ratio` as `None`.
4. WHEN the Reasoner suggests a directional command (`move_left` or `move_right`), THE Reasoner SHALL only emit that command if the per-side walkable ratio of the target Side is greater than or equal to the per-side walkable ratio of the current center Side.
5. IF every Side has a per-side walkable ratio below a configurable `min_lane_walkable_ratio` (default 0.10), THEN THE Reasoner SHALL emit `slow_down` rather than any directional move.

### Requirement 3: Multi-object reasoning

**User Story:** As a blind walker, I want one decision that accounts for everything in front of me, so that a person on the left and a pole on the right are both considered instead of the system fixating on one of them.

#### Acceptance Criteria

1. WHEN the Reasoner runs on a single frame, THE Reasoner SHALL produce exactly one GuidanceFacts object whose hazard summary includes every category whose per-side weighted pixel count exceeds `min_lane_walkable_ratio` × frame_area × 0.5 in any Side.
2. WHEN multiple categories are present in the same Side, THE Reasoner SHALL combine them into a single per-Side hazard description ranked by category priority from `navigation/reasoning/alerts.py` (`CATEGORY_PRIORITY`).
3. WHEN the Reasoner selects a final NavigationCommand, THE Reasoner SHALL pick the command using all three Sides together — Side selection logic SHALL NOT consider only the highest-priority hazard in isolation.
4. IF a Vision_STOP condition is met (CARE `hazard_detected` is true OR center-Side obstacle weighted ratio exceeds `hazard_obstacle_ratio`), THEN THE Reasoner SHALL emit `stop` regardless of any other Side state.
5. WHEN the Reasoner emits its decision, THE Reasoner SHALL include a structured `hazards_by_side` payload in GuidanceFacts so the Composer can describe each Side independently.

### Requirement 4: Approach direction detection

**User Story:** As a blind walker, I want to know whether something is crossing in front of me or planted in my path, so that I can react appropriately to a moving cyclist versus a stationary pole.

#### Acceptance Criteria

1. WHEN per-side counts are produced for a frame, THE Trend_Tracker SHALL push the per-category Side weight tuple `(left, center, right)` into a fixed-size ring buffer per category (default size 6).
2. WHEN the Trend_Tracker has at least 3 samples for a category, THE Trend_Tracker SHALL classify that category's approach direction as one of: `static`, `crossing_left_to_right`, `crossing_right_to_left`, `closing_in`, or `receding`.
3. WHEN the per-category centroid moves monotonically from the left Side toward the right Side across recent samples with a horizontal delta of at least 0.15 (in normalized [0,1] x), THE Trend_Tracker SHALL label the direction as `crossing_left_to_right`.
4. WHEN the per-category centroid moves monotonically from the right Side toward the left Side across recent samples with a horizontal delta of at least 0.15, THE Trend_Tracker SHALL label the direction as `crossing_right_to_left`.
5. WHEN the total weighted count for a category grows by at least the existing `growth_factor` (default 1.3) across the buffer, THE Trend_Tracker SHALL label the direction as `closing_in`.
6. WHEN the total weighted count for a category falls below half its earlier sample, THE Trend_Tracker SHALL label the direction as `receding`.
7. IF the per-category centroid horizontal delta is below 0.05 across the buffer, THEN THE Trend_Tracker SHALL label the direction as `static`.
8. WHEN approach direction labels are produced, THE Reasoner SHALL include them in GuidanceFacts under `approach_direction_by_category`.
9. WHEN the Trend_Tracker updates, THE Trend_Tracker SHALL perform `O(1)` work per category per frame (bounded by the ring buffer size).

### Requirement 5: Heuristic stairs/curb detector

**User Story:** As a blind walker, I want a heads-up when there is a step in my path, so that I can slow down before I trip on a curb or stair.

#### Acceptance Criteria

1. WHEN a frame is processed, THE Stairs_Detector SHALL examine the bottom 30% of the walkable region (intersection of the bottom 30% of the frame and walkable-class pixels) for horizontal-edge or luminance discontinuities.
2. WHEN the Stairs_Detector finds a horizontal gradient discontinuity that exceeds a configurable luminance-delta threshold (default 25 on an 8-bit grayscale) and spans at least 30% of the walkable region's width, THE Stairs_Detector SHALL emit a `stairs_or_curb` flag with a confidence in `[0.0, 1.0]` derived from edge strength.
3. WHEN the Stairs_Detector emits the flag, THE Reasoner SHALL include it in GuidanceFacts under `stairs_or_curb`.
4. WHEN GuidanceFacts contains `stairs_or_curb` with confidence at or above a configurable `stairs_min_confidence` (default 0.4), THE Composer SHALL produce a phrase semantically equivalent to "step ahead, slow down" using a Template_Library entry tagged `stairs_warning_low_conf`.
5. WHEN GuidanceFacts contains `stairs_or_curb` AND the active NavigationCommand is `stop`, THE Composer SHALL prepend the stairs warning to the stop phrase rather than replace the stop phrase.
6. WHEN the Stairs_Detector cannot find a walkable region (no walkable pixels in the bottom 30%), THE Stairs_Detector SHALL emit `stairs_or_curb=False` with confidence 0.0 and SHALL NOT raise an exception.
7. WHEN the Stairs_Detector runs on a frame, THE Stairs_Detector SHALL complete its analysis with `O(1)` cost relative to the existing per-frame inference time (i.e. operate on a downscaled view bounded by a constant size).

### Requirement 6: Approximate distance phrasing in feet

**User Story:** As a blind walker, I want approximate distances in feet that sound human, so that I can judge "how soon" without parsing a precise number.

#### Acceptance Criteria

1. WHEN the Composer formats a phrase that mentions distance, THE Composer SHALL select a Distance_Bucket (`immediate`, `near`, `mid`, `far`) using the current depth signal (segmentation-based proxy until metric depth lands).
2. WHEN the Distance_Bucket is `immediate`, THE Composer SHALL use a phrase semantically equivalent to "right in front of you" with no numeric distance.
3. WHEN the Distance_Bucket is `near`, `mid`, or `far`, THE Composer SHALL use the bucket's configured feet value with a hedge word from the set `{about, roughly, around}` (e.g. "about 30 feet ahead").
4. WHEN distance phrasing is produced, THE Composer SHALL never speak a precise non-rounded number (e.g. "32.4 feet"). Numeric values SHALL be drawn from the bucket configuration only.
5. WHERE the configuration provides a custom feet value per bucket, THE Composer SHALL use that value verbatim in templates without re-deriving from a depth scalar.
6. WHEN real metric depth replaces the 3-bucket proxy in a future change, THE Composer's phrase templates SHALL remain unchanged; only the bucketizer mapping `depth_meters → Distance_Bucket` SHALL be replaced.

### Requirement 7: Map + vision blended decisions

**User Story:** As a blind walker, I want one coherent sentence per cue that combines my route with what the camera sees, so that I am not confused by two voices contradicting each other.

#### Acceptance Criteria

1. WHEN both Map_Guidance and the Reasoner are active for a frame, THE Reasoner SHALL request the next Map_Guidance cue (turn direction and target bearing) and merge it into GuidanceFacts under `route_cue`.
2. WHEN GuidanceFacts contains both a `route_cue` and a non-stop vision command, THE Composer SHALL produce a single phrase that mentions both the route action and the vision context (e.g. "in about 30 feet, turn right; sidewalk is clear ahead").
3. IF GuidanceFacts contains a Vision_STOP, THEN THE Composer SHALL emit only the stop phrase and SHALL NOT mention the route cue in the same utterance.
4. IF Map_Guidance is unavailable (no route, no GPS, route fetch failed), THEN THE Composer SHALL produce a vision-only phrase using the same templates without throwing an error.
5. WHEN the Reasoner blends map and vision, THE Reasoner SHALL preserve the existing rule that vision STOP always wins (i.e. a Vision_STOP from CARE overrides any `move_left`/`move_right`/`go_forward` produced from Map_Guidance bearing).
6. WHEN map and vision agree on direction (e.g. both want `move_right`), THE Composer SHALL produce one combined phrase rather than two consecutive utterances.

### Requirement 8: Address textbox in phone client and `/set_destination` endpoint

**User Story:** As a blind walker, I want to type or paste an address into the phone client and have the system route me there, so that I do not have to edit `.env` files to change my destination.

#### Acceptance Criteria

1. WHEN the Phone_Client loads, THE Phone_Client SHALL render a single-line text input labeled "Destination" and a "Set" button positioned so it does not visually overlap the existing Start/Stop buttons.
2. WHEN the user submits a non-empty destination string, THE Phone_Client SHALL POST that string as form field `address` to `/set_destination` on the Phone_Server.
3. WHEN the Phone_Server receives a POST to `/set_destination` with a non-empty `address` field, THE Phone_Server SHALL geocode the address via the existing `navigation.maps.router.geocode_address` helper (Nominatim) and persist the resulting `(lat, lon)` as the active destination for the running NavigationInterpreter instance.
4. WHEN the Phone_Server successfully geocodes the address, THE Phone_Server SHALL respond with HTTP 200 and JSON `{"ok": true, "lat": <float>, "lon": <float>, "address": <resolved_display_name>}`.
5. IF the address cannot be geocoded (no result from Nominatim), THEN THE Phone_Server SHALL respond with HTTP 422 and JSON `{"ok": false, "error": "address_not_found"}`.
6. IF the request body is missing the `address` field or it is empty, THEN THE Phone_Server SHALL respond with HTTP 400 and JSON `{"ok": false, "error": "missing_address"}`.
7. WHEN a destination is set via `/set_destination`, THE NavigationInterpreter SHALL fetch a new OSRM walking route on the next frame that carries GPS coordinates and SHALL replace any prior Map_Guidance instance.
8. WHEN the user submits a new destination while navigation is running, THE Phone_Server SHALL apply the new destination without requiring a server restart.
9. WHERE the server has no current GPS (no recent frame with `lat`/`lon`), THE Phone_Server SHALL still accept and store the destination and SHALL defer route fetch until the first frame that carries GPS arrives.

### Requirement 9: Conversational voice with priority queue and per-tier cooldowns

**User Story:** As a blind walker, I want a steady conversational voice that warns me about danger first and chats about the scene only when nothing urgent is happening, so that I am informed without being overwhelmed.

#### Acceptance Criteria

1. WHEN the Voice_Queue receives utterances, THE Voice_Queue SHALL classify each utterance into exactly one Voice_Tier from the set `{vision_stop, directional_warning, map_turn, approach_alert, status_update}`.
2. WHEN a `vision_stop` utterance is enqueued, THE Voice_Queue SHALL interrupt any currently speaking utterance of any other tier and SHALL speak the stop phrase immediately.
3. WHEN a `directional_warning` utterance is enqueued AND the currently speaking utterance is a `status_update` or `approach_alert` or `map_turn`, THE Voice_Queue SHALL interrupt the current utterance and speak the directional warning.
4. WHEN a `map_turn` utterance is enqueued, THE Voice_Queue SHALL only speak it if at least `map_turn_min_interval_sec` (default 8.0 s) has elapsed since the last `map_turn` utterance.
5. WHEN an `approach_alert` utterance is enqueued, THE Voice_Queue SHALL only speak it if no `vision_stop`, `directional_warning`, or `map_turn` is currently being spoken or pending.
6. WHILE no other tier has spoken in the last `status_update_interval_sec` (default 10.0 s), THE Voice_Queue SHALL emit a single `status_update` describing the scene at most once per interval.
7. WHEN a `status_update` is in progress AND a higher-tier utterance arrives, THE Voice_Queue SHALL cancel the in-progress status update.
8. WHEN per-tier cooldowns are configured, THE Voice_Queue SHALL apply each tier's cooldown independently of every other tier.
9. WHEN the user has not enabled TTS (`tts_enabled=False`), THE Voice_Queue SHALL still log every utterance it would have spoken and SHALL NOT raise an error.

### Requirement 10: Template library for phrases

**User Story:** As a developer, I want spoken phrases pulled from a small data-driven template library with paraphrases, so that the voice does not feel repetitive and so that adding new phrases does not require code changes in the reasoning layer.

#### Acceptance Criteria

1. WHEN the Composer initializes, THE Composer SHALL load templates from a configuration file (default `config/phrases.yaml`) into an in-memory dictionary keyed by scenario tag.
2. WHEN the Composer needs a phrase for a scenario, THE Composer SHALL select uniformly at random from the 3–5 paraphrases registered under that scenario tag.
3. WHEN a template references a placeholder (e.g. `{distance_phrase}`, `{side}`, `{category}`), THE Composer SHALL substitute the placeholder with the value from GuidanceFacts using a small documented placeholder set.
4. WHERE the configuration file is missing or unreadable, THE Composer SHALL fall back to a built-in default template per scenario and SHALL log a warning once.
5. WHEN a scenario tag has fewer than 3 paraphrases registered, THE Composer SHALL still operate (selecting from whatever is present) and SHALL log a warning once on initialization listing the under-populated tags.
6. WHEN the same scenario fires twice in a short window (under 6 s), THE Composer SHALL avoid selecting the same paraphrase twice in a row when at least two paraphrases are configured.
7. WHEN a new scenario is added, THE developer SHALL only need to add an entry to the templates configuration file; no change in the reasoning layer SHALL be required to consume an additional paraphrase for an existing scenario.
8. WHEN the Composer renders a phrase, THE Composer SHALL produce a string with no unsubstituted placeholders. IF a required placeholder value is missing, THEN THE Composer SHALL fall back to a placeholder-free template for that scenario.

### Requirement 11: Pipeline performance budget

**User Story:** As a blind walker using a CPU-only laptop, I want guidance that stays real-time, so that warnings reach me before I walk into a hazard.

#### Acceptance Criteria

1. WHEN the full pipeline (Segmenter, Stairs_Detector, Trend_Tracker, Reasoner, Composer) processes a single frame on the reference machine (Intel Core Ultra 7 256V, no GPU, fast profile settings), THE pipeline SHALL complete in less than 100 ms on the median over a 100-frame benchmark.
2. WHEN the new spatial logic computes per-side counts and per-side walkable ratios, THE Segmenter SHALL perform that work in `O(H × W)` over the class map (already done once for the global counts) plus a constant amount of bookkeeping per Side.
3. WHEN the Reasoner produces GuidanceFacts from one SegmentationResult, THE Reasoner SHALL perform `O(1)` work in the size of the frame (only `O(C)` in the small set of tracked categories).
4. WHEN the Composer renders a phrase from GuidanceFacts, THE Composer SHALL perform `O(1)` work bounded by the number of paraphrases for the selected scenario.
5. WHEN benchmark mode is invoked from the CLI or a test, THE pipeline SHALL emit a per-stage timing record so regressions can be located.

### Requirement 12: Dependency and architecture constraints

**User Story:** As the maintainer, I want this feature to fit the current architecture without invalidating existing safety rules, so that the project stays small and the existing tests still pass.

#### Acceptance Criteria

1. WHEN this feature is implemented, THE System SHALL NOT add any new heavy runtime dependency (no LLM client library, no VLM, no torch version bump, no new GPU-only library) to `pyproject.toml`.
2. WHEN this feature is implemented, THE System SHALL NOT introduce any import from a higher Architecture_Rules layer into a lower one (utils, config, models, capture, perception SHALL NOT import from reasoning, maps, output, pipeline, or cli).
3. WHEN the Composer is added, THE Composer SHALL live in `navigation/reasoning/composer.py` (reasoning layer), and the output layer SHALL only consume rendered phrases produced by the Composer.
4. WHEN the Voice_Queue is added, THE Voice_Queue SHALL live in `navigation/output/` and SHALL NOT import from the pipeline layer.
5. WHEN this feature is implemented, THE System SHALL keep all 47 existing tests passing without modification of their assertions.
6. WHEN this feature adds new optional fields to SegmentationResult (`per_side_class_pixels`, `per_side_walkable_ratio`), THE fields SHALL default to `None` and SHALL NOT break any existing caller that constructs or consumes SegmentationResult.

### Requirement 13: Vision-STOP safety invariant

**User Story:** As a blind walker, I want a vision STOP to always win, so that no map prompt or chatty narration ever overrides a real safety warning.

#### Acceptance Criteria

1. WHEN the Reasoner detects a Vision_STOP condition for the current frame, THE Reasoner SHALL emit `NavigationCommand.STOP` regardless of any Map_Guidance suggestion.
2. WHEN a Vision_STOP utterance is enqueued in the Voice_Queue, THE Voice_Queue SHALL preempt every other tier (per Requirement 9.2).
3. WHEN both a route turn and a Vision_STOP would otherwise be spoken in the same frame, THE Composer SHALL emit only the stop phrase.
4. WHEN tested against any sequence of frames, THE System SHALL produce no spoken utterance that contradicts an active Vision_STOP within the same frame's voice schedule.

### Requirement 14: Stairs/curb is a heuristic, separable component

**User Story:** As a developer, I want the heuristic stairs/curb detector to be cleanly separable, so that a trained model can replace it later without rewriting the Composer or Reasoner.

#### Acceptance Criteria

1. WHEN the Stairs_Detector is invoked, THE Stairs_Detector SHALL accept a frame plus a SegmentationResult as inputs and SHALL return a small dataclass containing `flag: bool`, `confidence: float in [0,1]`, and `rationale: str`.
2. WHEN a future trained stairs model is introduced (separate spec), THE replacement SHALL satisfy the same input/output contract from criterion 1 so the Reasoner and Composer require no change.
3. WHEN the Stairs_Detector is disabled by configuration (e.g. `stairs_detector_enabled=False`), THE Stairs_Detector SHALL return `flag=False, confidence=0.0` and SHALL NOT contribute any phrase to the Composer.

### Requirement 15: Distance bucketizer is a swappable component

**User Story:** As a developer, I want the distance bucketizer to be the only thing that changes when real metric depth lands, so that the phrase templates and Composer are stable.

#### Acceptance Criteria

1. WHEN the Composer needs a Distance_Bucket, THE Composer SHALL call a single bucketizer function with a depth signal and SHALL receive a Distance_Bucket plus a feet value.
2. WHEN the depth signal is the current 3-bucket segmentation proxy, THE bucketizer SHALL map `obstacle_depth_m` values `<= 1.2`, `<= 2.2`, `<= 3.2`, and `> 3.2` to `immediate`, `near`, `mid`, `far` respectively (defaults; configurable).
3. WHEN real metric depth lands in a future spec, THE bucketizer's mapping function SHALL be the only required change to switch the system to metric depth.
4. WHEN the bucketizer is invoked with a `None` depth signal, THE bucketizer SHALL return the bucket configured as the conservative default (default `near`).
