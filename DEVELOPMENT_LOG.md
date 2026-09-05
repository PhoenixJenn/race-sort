# RaceSort Development Log

This document records the experimental history of RaceSort. `PROJECT_CONTEXT.md` is the source of truth for the current architecture and project state; this log preserves how and why the project reached that state.

## 2026-08-21 to 2026-08-23 — Initial Feasibility

### Problem Definition

RaceSort was defined as a local AI-assisted workflow for a motorsports photographer who may shoot approximately 1,000 photographs per hour for roughly 8 hours per day.

The existing workflow requires an assistant to inspect each photograph and type visible race numbers. Multi-vehicle photographs may need to be associated with multiple number folders.

Core requirement established: original photographs must never be destructively modified.

### Initial Development Hardware

- MacBook Pro
- Apple M3 Pro
- 18 GB unified memory
- macOS Sequoia 15.6
- approximately 100 GB free disk at project start

Decision: avoid unnecessarily large local models.

### Qwen3-VL Selection

RaceSort is fundamentally a visual recognition problem, so Qwen3-VL was selected for initial testing.

Initial `qwen3-vl:4b` behavior was excessively verbose/thinking-oriented.

Switched to:

```text
qwen3-vl:4b-instruct
```

### Image Attachment Lesson

Typing or mentioning a local image path does not reliably provide the image to the vision model.

The image must be explicitly passed through the Ollama message `images` field.

Changing Terminal working directory alone does not give a model visual access to files.

### Whole-Image Qwen Test

Known obvious numbers in GGBM0021 included 49 and 706.

Whole-scene result:

```text
49 70
```

49 was correct; 706 was misread as 70.

Conclusion: whole-image multi-target recognition is not reliable enough by itself.

### Full-Resolution Context Failure

A focused full-resolution request produced:

```text
request (4113 tokens) exceeds the available context size (4096 tokens)
n_prompt_tokens: 4113
n_ctx: 4096
```

Conclusion: do not routinely send full camera-resolution originals to the VLM.

### Resize Experiment

A resized/focused GGBM0021 test returned:

```text
706
```

correctly.

Focused tests on GGBM0022 returned:

```text
215
49
```

correctly.

Working hypothesis established: isolate a vehicle first, then perform number recognition on that crop.

## 2026-08-23 — Python Integration

Python environment:

```text
Python 3.12.4
```

Initial virtual environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ollama pillow
```

`test_qwen.py` proved:

```text
Python → Pillow → Ollama → Qwen3-VL → vision result
```

Successful result:

```text
Created resized image: test-photos/GGBM0021-small.jpg
Qwen response:
706
```

Milestone: local Python-controlled Qwen vision proof of concept complete.

## 2026-08-23 — Phase 2A: Automatic Vehicle Detection

Installed PyTorch/Transformers detection dependencies.

Initial DETR load failed because `timm` was missing.

Installed:

```bash
pip install timm
```

Lesson: `timm` is an explicit dependency of the current DETR/Transformers setup.

Vehicle detector:

```text
facebook/detr-resnet-50
```

GGBM0021 result:

```text
Motorcycle 1: confidence=0.99
Motorcycle 2: confidence=1.00
Motorcycle 3: confidence=0.98
Motorcycles detected: 3
```

Visual inspection confirmed the three boxes enclosed the useful/visible motorcycles.

The photograph contained additional heavily obscured/background motorcycles that DETR omitted.

This led to a key metric:

> RaceSort should optimize for sortable-vehicle recall, not detection of every tiny/obscured vehicle fragment.

Milestone: automatic useful-motorcycle detection proven.

## Phase 2B / 2C — Crop-Based Number Recognition and Batch Work

DETR crops were sent independently to Qwen.

The project expanded from single-image proof-of-concept work into batch processing and review experiments.

The 19-photo benchmark eventually produced 33 vehicle crops and became the main regression dataset.

Human review remained a required part of the architecture rather than treating model output as infallible.

## 2026-08-23 — Phase 2D: Visual Identity and Evidence Fusion

### DINOv2 Added

Model:

```text
facebook/dinov2-small
```

Purpose:

- create local embeddings for vehicle crops
- compare REVIEW crops with confirmed registry sightings
- provide independent vehicle-identity evidence

Initial similarities included:

```text
#52 vs #52 → 0.9654
#869 vs #869 → 0.8473
#52 vs #869 → 0.7356
confirmed #54 vs blurry #54 → 0.7569
confirmed #54 vs another #54 → 0.9537
```

Conclusion:

- DINO similarity is useful
- viewpoint matters
- one universal threshold is not automatically safe
- supporting evidence and ranking matter

### `match_review.py`

Created review matching against confirmed registry identities.

The script preserved all confirmed identities in JSON and reported best/mean similarity and supporting sightings.

Finding: DINO is useful for candidate generation/rejection, but similarity alone should not automatically assign identity.

### `evaluate_candidates.py`

Added evidence fusion.

Evidence included:

- profile number
- verification number
- DINO rank
- DINO best similarity
- DINO mean similarity
- make compatibility
- vehicle colors
- rider leathers
- helmet colors

Candidate classes:

```text
STRONG_CANDIDATE
POSSIBLE_CANDIDATE
CONFLICT
INSUFFICIENT_EVIDENCE
```

Important #54 outcomes included:

```text
GGBM0017 → verification 54 + DINO #1 → STRONG #54
GGBM0005 → verification 54 + compatible DINO evidence → POSSIBLE #54
```

### Photo-Level Assignments

`build_photo_assignments.py` preserved provenance:

```text
READ
INFERRED
REVIEW
```

Policy at that stage:

```text
CONFIRMED first-pass → READ
STRONG_CANDIDATE → INFERRED
POSSIBLE / CONFLICT / INSUFFICIENT → REVIEW
```

19-photo benchmark at that checkpoint:

```text
7 fully resolved photos
12 photos needing review
14 photos with at least one direct READ
1 second-pass INFERRED assignment
```

Known inferred recovery: blurry #54.

## Race Number Data Model Correction

A race number can legitimately be:

```text
007
```

Therefore race numbers must not be treated as integers.

New rules:

- opaque string identifiers
- preserve leading zeros
- `"007"` != `"7"`
- current racers may use only digits, but letters must not break the system
- future examples such as `54A` and `A12` must remain valid

Normalization accepts:

```text
[A-Z0-9]{1,6}
```

Prompts were updated to preserve leading zeros and character order.

## Offline Requirement

RaceSort must operate without internet access at the racetrack.

Current local model categories:

- Qwen3-VL through Ollama
- DETR
- DINOv2

Production code should use cached/local models and offline loading rather than depending on event-time downloads.

## Phase 3A — Performance Instrumentation

The initial 19-photo batch was too slow, so timing instrumentation was added before optimizing.

Measured components included:

- DETR
- Qwen profile
- Qwen verification
- per-vehicle time
- per-photo time
- total batch time

### Rich Pipeline Baseline

```text
Total batch time: 313.08s
Photos processed: 19
Vehicles processed: 33
DETR total: 10.55s
Qwen profile total: 280.37s
Qwen verification total: 16.58s
Average seconds/photo: 16.48
Projected 1,000-photo time: 274.6 minutes
Average profile call: 8.50s
Average verification call: 0.50s
```

Finding: Qwen rich profiling overwhelmingly dominated runtime.

## Fast Number Experiments

A faster two-pass number-reading approach was tested.

One full 33-vehicle result:

```text
Vehicles tested: 33
Confirmed: 14
Review: 19
Pass A total: 10.31s
Pass B total: 31.94s
Total elapsed: 42.26s
Average Pass A: 0.31s
Average Pass B: 0.97s
Average two-pass time/vehicle: 1.28s
```

The experiment demonstrated that focused short prompts could be much faster than full rich profiles, although behavior varied between runs/model state.

### Integrated Fast + Rich Batch

A later integrated run produced:

```text
Photos processed: 19
Vehicles processed: 33
Fast-confirmed vehicles: 15
Vehicles for REVIEW: 18
Rich profiles generated: 27
Total batch time: 254.26s
DETR total: 11.81s
Fast Pass B total: 124.03s
Fast Pass A total: 10.64s
Rich profile total: 101.35s
Average Fast Pass B: 3.76s
Average Fast Pass A: 0.32s
Average fast two-pass: 4.08s
Average vehicle total: 7.17s
Average seconds/photo: 13.38
Projected 1,000-photo time: 223.0 minutes
```

Conclusion: still much too slow. Reducing Qwen calls became a priority.

## RapidOCR Experiment

RapidOCR was tested despite OCR having originally been postponed.

33-crop baseline:

```text
Vehicles tested: 33
OCR total: 7.17s
Batch elapsed: 7.17s
Average OCR time/vehicle: 0.217s
```

Conclusion: OCR is cheap enough to be valuable as a routing stage.

This reversed the earlier “postpone OCR” working assumption based on measured evidence.

## OCR + Conditional Qwen

OCR candidates were sent to Qwen for verification.

Result:

```text
Vehicles tested: 33
Qwen calls made: 13
Confirmed: 10
Review: 23
OCR total: 8.48s
Qwen total: 80.79s
Batch elapsed: 89.28s
Average OCR time/vehicle: 0.257s
Average total time/vehicle: 2.71s
Average Qwen call: 6.21s
```

One important correct case:

```text
GGBM0008 / motorcycle-01
OCR candidate: 0
Qwen verification: 0
Decision: CONFIRMED
```

A race number of `0` is valid and must not be treated as missing/falsey data.

## Human Validation Webpage

A dynamic offline HTML reviewer was created so automated results could be viewed alongside each vehicle crop.

The reviewer was expanded to expose increasingly verbose evidence:

- current number result
- OCR candidate(s)
- Qwen output
- vehicle metadata
- registry data
- DINO matches
- candidate evaluation
- assignment/provenance information

This allowed visual identification of failure modes that were hidden in CSV-only review.

Important observation: several crops previously identified correctly were becoming UNKNOWN in the OCR-focused route.

Another observation: some detected motorcycles were too blurry or too minor to justify additional expensive processing.

## Non-Primary / Garbage Filter Experiment

Human validation distinguished useful primary/secondary motorcycles from background/non-primary crops.

The best zero-false-filter rule on the labeled data was:

```text
vehicles_in_photo > 1
AND relative_area < 0.20
AND relative_sharpness < 0.45
```

Initial report described 2 “safely filtered” labeled cases.

When applied across the full 33-crop routing batch, it filtered 7 crops.

Human inspection confirmed those 7 looked accurate, so this was not considered a problem.

Current outcome name:

```text
FILTERED_NON_PRIMARY
```

## Sellability / Blur Classification

Human review revealed a separate concept from primary/non-primary status:

A crop can be a significant motorcycle in the frame but still be too blurry to sell.

The reviewer was expanded with:

```text
SELLABLE
BORDERLINE
TOO_BLURRY
```

These labels are independent of:

```text
PRIMARY
SECONDARY
NON_PRIMARY
```

### Human Quality Labels

At one checkpoint:

```text
21 SELLABLE
7 TOO_BLURRY
1 BORDERLINE
4 unlabeled
```

Two problematic blurry crops:

```text
GGBM0005 motorcycle-01
sharpness ≈ 121.7
human: TOO_BLURRY

GGBM0006 motorcycle-01
sharpness ≈ 108.9
human: TOO_BLURRY
```

These crops had caused direct-Qwen hallucinations (`64` and `68`) despite being too blurry to sell.

Lowest observed SELLABLE sharpness examples were above roughly 218.

Threshold testing showed `<200` could catch four TOO_BLURRY examples with zero labeled SELLABLE rejects, but `<150` caught the same four in the labeled subset and was more conservative.

Decision for the next routing experiment:

```text
sharpness < 150
→ FILTERED_TOO_BLURRY
```

## Qwen Hallucination Finding

Human validation established that direct Qwen reads cannot be automatically trusted.

Examples of clear numberless Aprilia motorcycles produced plausible-looking race-number guesses such as:

```text
81
13
```

and later:

```text
83
81
```

These were not quality failures; the images were sellable/clear but did not show a valid visible race number.

Architectural rule established:

> A direct Qwen number read is a candidate, not a confirmation.

## Routing Pipeline V1

A combined routing experiment used:

```text
garbage filter
→ OCR
→ Qwen verification when OCR produced candidates
→ direct Qwen read when OCR produced nothing
```

The run exposed:

- direct Qwen fallback could recover numbers OCR missed
- single direct Qwen reads could hallucinate plausible numbers
- filtered count across all rows differed from the earlier labeled-only “safely filtered” count

This motivated Routing V2.

## Routing Pipeline V2

Changes:

1. non-primary filter first
2. absolute blur filter second
3. RapidOCR
4. OCR candidate → Qwen verification
5. rejected OCR → direct Qwen fallback
6. empty OCR → direct Qwen read
7. direct Qwen numbers become `QWEN_CANDIDATE`, not `CONFIRMED`

Result:

```text
Vehicle crops: 33
Filtered non-primary: 7
Filtered too blurry: 2
Filtered total: 9
Processed further: 24
OCR candidate cases: 13
OCR empty cases: 11
Qwen VERIFY calls: 13
Qwen DIRECT calls: 14
Total Qwen calls: 27
CONFIRMED: 10
QWEN_CANDIDATE: 9
REVIEW: 5
OCR total: 7.73s
Qwen VERIFY total: 80.46s
Qwen DIRECT total: 55.10s
Batch elapsed: 143.31s
Average total time / processed vehicle: 5.97s
```

### Key V2 Outcomes

The two blurry problem crops were filtered before Qwen, preventing the earlier `64` / `68` hallucinations.

Bad OCR recovery worked:

```text
GGBM0017 motorcycle-01
OCR: C42A
Qwen verification: UNKNOWN
direct Qwen: 54
→ QWEN_CANDIDATE
```

and:

```text
GGBM0018 motorcycle-01
OCR: 122
Qwen verification: UNKNOWN
direct Qwen: 721
→ QWEN_CANDIDATE
```

Numberless Aprilia hallucinations were contained as candidates rather than confirmations.

## Candidate Resolution Experiment

`test_candidate_resolution.py` evaluated the 9 `QWEN_CANDIDATE` rows using existing registry/DINO/fusion evidence.

Initial result:

```text
Qwen candidates evaluated: 9
CORROBORATED: 1
SUPPORTED_REVIEW: 0
CONFLICTING: 2
UNSUPPORTED: 6
```

The one CORROBORATED case was #54 with strong independent evidence.

The two CONFLICTING cases were useful because they included numberless Aprilia hallucinations.

Inspection showed the six UNSUPPORTED cases were misleadingly named: their candidate numbers already existed in the registry, but those crops lacked matching independent DINO/fusion evidence in the old files.

Architectural lesson:

> Missing evidence is not the same as conflicting evidence.

A new state was added:

```text
KNOWN_NUMBER_REVIEW
```

Updated result:

```text
Qwen candidates evaluated: 9
CORROBORATED: 1
SUPPORTED_REVIEW: 0
KNOWN_NUMBER_REVIEW: 6
CONFLICTING: 2
UNSUPPORTED: 0
```

## DINO Candidate Coverage Experiment

A new script compared known Qwen candidates directly with confirmed registry sightings of the same candidate number.

### First Run Problem: Self-Matches

The initial implementation accidentally allowed a candidate crop to compare against itself when that same crop already existed in the registry.

This produced meaningless similarities around:

```text
1.0000
```

Lesson:

> A DINO self-match is not independent evidence and must never be used for identity corroboration.

The script was changed to skip any reference path equal to the candidate crop path.

### Corrected DINO Coverage

Corrected run:

```text
Known Qwen candidates tested: 7
Embeddings computed: 12
Batch elapsed: 0.54s
```

Independent evidence:

```text
GGBM0005 bike 2 → #52
best 0.9654
mean 0.9594

GGBM0009 bike 2 → #721
best 0.9149
mean 0.8821

GGBM0013 bike 1 → #98
no independent reference available

GGBM0015 bike 1 → #869
best 0.9387
mean 0.8974

GGBM0016 bike 1 → #52
best 0.9534
mean 0.9352

GGBM0017 bike 1 → #54
best 0.9537
mean 0.9537

GGBM0018 bike 1 → #721
best 0.9149
mean 0.8960
```

Six of seven known Qwen candidates with an independent reference had best DINO similarity above 0.90.

The #98 case did not fail; it simply had no second independent confirmed #98 reference.

Experimental candidate promotion rule proposed:

```text
Qwen candidate matches existing registry number
AND best independent DINO similarity >= 0.90
→ eligible for CORROBORATED
```

This threshold remains provisional because the benchmark is small.

## Cross-Platform Deployment Requirement Added

Photographer reference PC:

```text
Intel Core i7-8700
6 cores / 12 threads
3.2 GHz base / 4.6 GHz turbo
48 GB RAM
NVIDIA GTX 1050 Ti
4 GB VRAM
```

Requirement:

RaceSort must remain one application/codebase for both Apple Silicon macOS and Windows/NVIDIA hardware.

Future configuration should externalize hardware-specific choices rather than forking application logic.

Planned profile concept:

```text
auto
mac_apple_silicon
windows_nvidia
CPU fallback
```

Likely configurable items:

- model names
- PyTorch device
- detector device
- DINO device
- Ollama/vision model
- worker counts
- OCR concurrency
- embedding concurrency
- maximum vision workers
- paths

The GTX 1050 Ti's 4 GB VRAM is expected to constrain GPU model choice/concurrency.

This configuration layer should be built after the current routing pipeline is consolidated and benchmarked.

## Parallelism

Parallel processing was identified as a future optimization.

Likely safe/cheap candidates:

- preprocessing
- image-quality metrics
- OCR
- DINO embeddings
- file operations

Qwen concurrency must be measured rather than assumed. The Mac's unified memory and the Windows PC's 4 GB VRAM may require different concurrency limits.

## Current Checkpoint

The project now has experimentally validated pieces for:

- DETR vehicle detection
- crop-based processing
- non-primary filtering
- blur/sellability filtering
- RapidOCR
- Qwen candidate verification
- direct Qwen fallback
- safe Qwen candidate state
- DINO embeddings
- registry matching
- evidence fusion
- candidate resolution
- human validation
- performance instrumentation
- opaque race-number strings with leading-zero preservation
- offline operation requirements

The project has reached the point where additional isolated recognition experiments have diminishing value.

## Next Milestone

Consolidate the validated components into the main RaceSort processing pipeline.

Run the existing 19-photo / 33-crop regression set end-to-end and report:

```text
photos processed
vehicles detected
FILTERED_NON_PRIMARY
FILTERED_TOO_BLURRY
OCR candidate cases
OCR empty cases
Qwen verification calls
Qwen direct calls
CONFIRMED
CORROBORATED
KNOWN_NUMBER_REVIEW
CONFLICTING / REVIEW
total review workload
DETR time
OCR time
Qwen time
DINO time
total batch time
projected 1,000-photo time
```

Only after correctness is preserved in the consolidated pipeline:

1. benchmark parallelism
2. create cross-platform hardware/config profiles
3. test clean Mac startup from scratch
4. deploy/test on the photographer's Windows PC
5. verify fully offline event operation
6. continue toward production export/copy workflow and polished UI

## 2026-08-24 — Consolidated Regression Caught Anchored Verification Failure

The validated routing, quality filters, and independent DINO resolution were consolidated into `test_pipeline.py`. A new read-only regression checker was added at `regression/check_pipeline_results.py`.

One repeated regression run exposed a false automatic confirmation:

```text
GGBM0018 motorcycle-01
RapidOCR: 122
candidate-anchored Qwen verification: 122
actual visible number: 721
```

This demonstrated that OCR plus a Qwen prompt containing the OCR candidate is not two fully independent signals. The regression checker correctly failed rather than accepting the lower review workload as an improvement.

The routing rule was made more conservative:

```text
OCR candidate
+ anchored Qwen verification
+ unanchored direct Qwen read
→ CONFIRMED only when all three agree exactly
```

Otherwise, a nonempty direct read remains `QWEN_CANDIDATE`, and an unreadable direct result becomes `REVIEW`.

The corrected 19-photo / 33-crop run produced:

```text
CONFIRMED: 10
CORROBORATED: 4
KNOWN_NUMBER_REVIEW: 2
UNSUPPORTED: 3
REVIEW: 5
human-review workload: 10
Qwen verification calls: 13
Qwen direct calls: 24
regression checks: 89 passed, 0 failed
```

The same run completed in 51.22 seconds, but this speed must not yet be attributed to a code optimization because Qwen/Ollama latency varied substantially between otherwise similar runs.

## 2026-08-24 — Skip Anchored Verification When OCR and Direct Read Disagree

The three-signal confirmation policy requires OCR, an unanchored direct read, and anchored verification to agree exactly. Therefore, anchored verification cannot promote a crop when OCR and the direct read already disagree.

The hardware-agnostic routing order was changed to:

```text
RapidOCR
→ unanchored direct Qwen read
→ compare OCR and direct result
→ run anchored verification only when they already agree
```

This reduced anchored verification calls from 13 to approximately 9–10 while keeping 24 direct safety reads.

Regression results remained safe:

```text
hard regression failures: 0
human-review workload: 10–11
```

Warm timing did not establish a measurable speedup:

```text
previous warm runs: 51.24s, 57.08s
optimized warm runs: 53.39s, 62.49s
```

Qwen timing variance was larger than the time saved by three verification calls. The change was retained because it removes logically unnecessary model work without weakening confirmation safety, but it must not be represented as a proven throughput improvement.

## 2026-08-25 — RapidOCR One-Worker vs Two-Worker Benchmark

An isolated benchmark was added at:

```text
experiments/benchmark_ocr_workers.py
```

The experiment used the 24 current crops that passed the validated quality filters. Each executor thread owned a separate persistent RapidOCR engine; no engine instance was shared concurrently. Workers were warmed before timing.

Three timed rounds per mode produced:

```text
serial:      6.424s, 5.816s, 5.840s
two workers: 5.033s, 5.021s, 5.052s

serial median:      5.840s
two-worker median:  5.033s
median speedup:     1.16x
```

Raw OCR texts and normalized candidate lists were identical across every serial and two-worker round.

Conclusion:

- two independent OCR workers produced a repeatable but modest improvement of approximately 0.81 seconds across 24 crops;
- ONNX Runtime already performs internal parallel work, so two Python workers did not approach a 2x speedup;
- integrating this result would require restructuring the current photo-by-photo pipeline to stage OCR work across crops;
- the expected end-to-end saving is currently too small to justify that larger pipeline change.

The result is preserved as evidence, but OCR concurrency is not yet integrated into the main pipeline.

## 2026-08-25 — Qwen Serial vs Two-Call Concurrency Benchmark

An isolated benchmark was added at:

```text
experiments/benchmark_qwen_concurrency.py
```

The experiment used the main pipeline's unanchored direct-read prompt on four fixed crops:

- clear #49;
- valid race number `0`;
- recovery case #54;
- a clear but numberless motorcycle expected to return `UNKNOWN`.

The first benchmark version warmed only one crop, which gave the later two-call mode an unfair warm-state advantage. That preliminary `13.79x` headline was rejected as invalid. The benchmark was corrected to:

- warm every fixed crop before timing;
- run serial first in odd rounds;
- run two-call mode first in even rounds;
- preserve raw responses, normalized identifiers, per-call timing, execution order, API errors, and Ollama model state.

Corrected results:

```text
Round 1
serial first: 0.65s
two-call:     0.82s

Round 2
two-call first: 0.67s
serial:         0.68s

serial median:   0.67s
two-call median: 0.75s
relative speed:  0.89x
API errors:      0
```

All three readable identifiers remained correct in every call:

```text
49
0
54
```

The numberless motorcycle hallucinated a different candidate depending on the run and scheduling:

```text
serial:   86, 81
two-call: 38, 83
```

Conclusion:

- two simultaneous Qwen calls were approximately 11% slower than serial calls on the current development setup;
- concurrent scheduling changed nondeterministic hallucinated output;
- the numberless outputs remain safely contained by the main pipeline's candidate/review policy;
- Qwen concurrency must not be integrated into the current main pipeline;
- Qwen remains serialized by default.

## 2026-08-25 — Number-Visibility Gate Rejected

An isolated three-state visibility experiment was added at:

```text
experiments/benchmark_number_visibility.py
```

The proposed gate asked Qwen to return exactly one of:

```text
VISIBLE_NUMBER
NO_NUMBER_VISIBLE
UNCLEAR
```

Only `NO_NUMBER_VISIBLE` would have been eligible to bypass later number-reading work. `UNCLEAR` was designed to continue through normal routing.

The experiment used all 24 processable crops with available human validation:

```text
CLEAR: 16
NO_NUMBER_VISIBLE: 5
NOT_READABLE: 1
UNLABELED: 2
```

One round produced:

```text
VISIBLE_NUMBER: 19
NO_NUMBER_VISIBLE: 5

clear-number false rejections: 0
unsafe uncertain/unlabeled rejections: 2
numberless caught: 3/5
numberless recall: 60.0%
API errors: 0
total time: 146.61s
median call time: 6.50s
```

Unsafe results:

- GGBM0013 vehicle 2 was human-labeled `NOT_READABLE` but the gate returned `NO_NUMBER_VISIBLE`;
- GGBM0018 vehicle 2 was unlabeled but the gate returned `NO_NUMBER_VISIBLE`.

Missed confirmed numberless crops:

- GGBM0012 vehicle 1 returned `VISIBLE_NUMBER`;
- GGBM0020 vehicle 2 returned `VISIBLE_NUMBER`.

Conclusion:

- the gate was too slow to be a cheap filter;
- it caught only three of five confirmed numberless crops;
- it made two unsafe rejection decisions on uncertain evidence;
- no second round was justified under the predeclared stop rule;
- the visibility gate must not be integrated into the main pipeline.

## 2026-09-04 — Opt-In Merged-Motorcycle Recovery Validated

Human review found that `GGBM0082.JPG` contains motorcycles `869` and
`215`, while the normal DETR threshold produced one high-confidence box
covering both motorcycles.

An isolated 158-photo sweep tested whether lower-confidence DETR boxes could
conservatively split a high-confidence merged parent. The initial geometric
rule produced three false proposals. Adding both of these requirements removed
the observed false proposals:

- child boxes must have an area balance of at least `0.50`;
- their horizontal centers must be separated by at least `0.33` of the parent
  width.

At the selected child threshold of `0.275`, the refined rule proposed splits
for only two photos:

```text
GGBM0021.JPG: 4 resolved motorcycles
GGBM0082.JPG: 2 resolved motorcycles
```

Both proposals were visually valid. A complete two-photo pipeline run then
produced:

```text
GGBM0021: two baseline crops and two merged-box child crops
GGBM0082: merged-box child #869, UNSUPPORTED
GGBM0082: merged-box child #215, UNSUPPORTED
```

The identifiers `869` and `215` were preserved as strings. `UNSUPPORTED`
remains a human-review outcome, so the recovered numbers were not promoted to
automatic assignments without corroborating evidence.

Implementation status:

- the recovery logic is available in `test_pipeline.py` only when
  `RACESORT_ENABLE_MERGED_BOX_SPLIT=1`;
- default `0.70` detection behavior is unchanged;
- each result records `detection_source` as `baseline` or
  `merged_box_child`;
- detector-report and pipeline-output regression checks both pass;
- the established 19-photo regression has zero failures.

The next validation step is a full-dataset opt-in run followed by comparison
of detection counts, review workload, and human-validated false splits before
considering whether to enable the behavior by default.

## 2026-09-04 — Opt-In Content-Addressed Qwen Cache Validated

The 158-photo merged-box run took `1421.74s` (`23m 42s`). Qwen direct
recognition accounted for `1197.85s`, or approximately 84% of the complete
runtime. The pipeline made 244 direct calls with a median of `5.03s` and a
mean of `4.91s`; the cost was steady per-crop inference rather than a single
startup delay.

An opt-in persistent cache was added around raw Qwen responses. A cache key
includes:

- SHA-256 of the generated crop contents;
- the exact prompt;
- the configured Qwen model name;
- a cache schema version.

The cached raw response still passes through the existing identifier
normalization and routing logic. Failed API calls are not cached, writes are
atomic, and `.racesort-cache/` is excluded from Git. Caching is disabled by
default and enabled with:

```text
RACESORT_ENABLE_QWEN_CACHE=1
```

A two-pass `GGBM0082.JPG` validation produced:

```text
Cold run
cache hits:       0
cache misses:     2
Qwen time:        14.25s
total time:       16.08s

Warm run
cache hits:       2
cache misses:     0
Qwen time:        0.00s
total time:       1.97s
```

The cold and warm runs produced identical recognition evidence, race-number
strings, and routing decisions. The cache improves repeated experiments and
restartability; it does not reduce first-pass inference time on previously
unseen crops.

## 2026-09-04 — Golden DINO Plus OCR Pre-Qwen Benchmark

A read-only benchmark evaluated whether a human-validated golden-record
registry could safely bypass direct Qwen reads on fresh crops. The experiment
used the latest 311-row human-validation export and selected the sharpest
qualifying reference for each race-number string:

```text
golden identities:              27
non-golden processable queries: 216
```

Golden references required human `CORRECT`, `CLEAR`, `SELLABLE`, a
`PRIMARY`/`SECONDARY` role, sharpness of at least `500`, and a non-empty
string identifier. Golden crops were excluded from their own evaluation.

Allowing DINO to agree with any identifier in a multi-candidate OCR result was
unsafe at the existing `0.90` similarity threshold:

```text
proposed Qwen skips: 16
human-evaluated:     16
incorrect:            1
precision:          93.8%
```

The error was `GGBM0066.JPG` vehicle 4. Human ground truth was `706`, OCR
contained `P7`, `706`, and `866`, and DINO incorrectly selected `866` at
similarity `0.9112`.

Requiring OCR to contain exactly one candidate removed all evaluated errors:

```text
threshold:            0.90
proposed Qwen skips:     12
human-evaluated:         12
incorrect:                0
precision:           100.0%
```

Conclusion:

- multi-candidate OCR/DINO agreement must not bypass Qwen;
- single-candidate OCR plus DINO at `0.90` is promising but is supported by
  only 12 evaluated cases;
- the route would avoid only about 5.6% of calls in this dataset, roughly one
  minute at the measured Qwen rate;
- do not integrate automatic promotion from this benchmark;
- prioritize an input-resolution benchmark because image preprocessing may
  reduce the cost of every fresh Qwen call rather than a small subset.

## 2026-09-04 — Qwen Input-Resolution Benchmark

Fifty previously unseen photographs were kept outside Git and processed with
DETR only. The detector produced 71 non-destructive motorcycle crops. Human
review labeled all 71 crops:

```text
NUMBER: 47
NONE:   16
UNSURE:  8
```

A first balanced stress test compared maximum crop dimensions of `1500`,
`1024`, and `768` pixels on 12 crops. It included the lowest-sharpness readable
crops that supported all three sizes and the sharpest confirmed-numberless
crops:

```text
1500: 91.7% accuracy, 5.61s median
1024: 100.0% accuracy, 3.27s median
 768: 100.0% accuracy, 3.22s median
```

The only stress-test error occurred at 1500 pixels: human label `181` was read
as `187`. All five numberless cases returned `UNKNOWN` at every size. Because
768 saved only about 0.05 seconds per call compared with 1024 while discarding
more detail, 1024 was selected for broader validation.

The complete 1024-pixel validation then evaluated every definitive human
answer:

```text
cases:                 63
exact readable number: 45/47
numberless UNKNOWN:     15/16
overall accuracy:       95.2%
median call time:       3.25s
```

Observed errors:

- `GGBM1130` vehicle 1: `00` read as `99`;
- `GGBM1138` vehicle 1: `128` read as `T28`;
- `GGBM1159` vehicle 1: confirmed numberless crop hallucinated as `91`.

Conclusion:

- 1024 pixels is the provisional performance choice for Qwen candidate reads;
- it reduced mean call time from about 5.64 seconds at 1500 to about 3.30
  seconds in the balanced comparison;
- a Qwen read at 1024 must not automatically confirm an identity by itself;
- independent OCR, registry, DINO, metadata, or human evidence remains
  necessary for safe promotion;
- do not change the main pipeline default until the configurable resize path
  passes the established regression suite.

## 2026-09-04 — Human-Confirmed Registry DINO Pilot

A read-only experiment simulated the proposed checkpoint between a fast first
pass and a registry-assisted second pass. From the 50-photo resolution dataset,
the experiment selected one human-confirmed golden crop per race-number string,
preferring `CLEAR` readability and then the sharpest crop. Every additional
sighting was compared against all confirmed registry identities.

The dataset contained:

```text
confirmed registry identities: 33
identities with later sightings: 12
later-sighting query crops:      14
self-matches:                     0
```

DINOv2 Small completed the local CPU run in 2.92 seconds. Ranking results were:

```text
correct identity ranked first: 10/14
correct identity in top three:  11/14
```

Threshold behavior for the top-ranked suggestion was:

```text
threshold  accepted  errors  precision
0.75          14       4       71.4%
0.80          13       3       76.9%
0.85          11       1       90.9%
0.90           9       0      100.0%
0.92           7       0      100.0%
0.95           3       0      100.0%
```

The four incorrect top-ranked results were later sightings of `69`, `10`,
`10`, and `957`. Their top similarities were all below `0.90`. A correct `45`
match scored `0.8978`, demonstrating the expected coverage tradeoff near the
threshold.

Conclusion:

- a human-confirmed first-pass registry can provide useful, extremely cheap
  suggestions for later sightings;
- the existing provisional `0.90` threshold separated all observed correct
  and incorrect top matches in this small pilot;
- 14 queries are not enough to declare automatic promotion safe;
- lower-scoring and viewpoint-difficult crops need additional OCR, metadata,
  multiple-reference, or human evidence;
- the next experiment should measure whether structured vehicle/rider metadata
  improves ranking for the unresolved cases without overriding contradictory
  race-number evidence.

## 2026-09-04 — Combined-Set Registry Pooling Was Unsafe

The confirmed-registry experiment was extended to accept multiple validation
CSVs without copying or moving photographs. The 311-row earlier validation set
and the 71-row resolution set initially treated each number as if it had one
canonical visual identity and produced:

```text
validation rows:             382
confirmed registry identities: 51
later-sighting query crops:  151
query identities:             37
self-matches:                  0
```

The larger result did not preserve the apparent safety of the 14-query pilot:

```text
correct identity ranked first: 45/151
correct identity in top three:  54/151

threshold  accepted  errors  precision
0.90          44      15       65.9%
0.92          20       3       85.0%
0.95           3       0      100.0%
```

The earlier photo collection had been assembled from several source folders,
identified by `_1`, `_2`, `_3`, and unsuffixed filename groups. Pooling those
groups also exposed an invalid assumption in the experiment: a race number is
not a unique motorcycle/rider identity. Multiple distinct motorcycles may
legitimately share one number, even within the same event. One canonical visual
reference per number therefore cannot represent all valid appearances. Race
numbers may also be reused across events, and one reference remains
viewpoint-sensitive even when the physical motorcycle is the same.

Conclusion:

- more photographs are not automatically more valid identity evidence;
- every registry must have an explicit event boundary;
- each race-number string must support multiple motorcycle/rider variants;
- each variant should support multiple confirmed viewpoint references;
- cross-event records must not be pooled solely because their race-number
  strings match;
- the reported ranking evaluated same-number retrieval, not unique physical
  identity recognition;
- before spending Qwen calls on metadata, test multiple independently
  confirmed variants and viewpoints per number inside a known event or batch;
- DINO similarity alone must remain suggestion evidence, not an automatic
  assignment rule.

## 2026-09-04 — Number-Blind Visual Metadata Pilot

A metadata-only Qwen experiment was created for event-registry variants. The
human race-number label is retained only for later scoring and is never placed
in the prompt. Source crops are copied non-destructively to a maximum dimension
of 1024 pixels. The profile records motorcycle colors/patterns, rider leathers,
helmet appearance, number-plate background, view, and occlusion without
transcribing the number.

The first prompt was overly restrictive. On eight crops for numbers `10`,
`69`, and `957`, only two profiles contained at least three useful visual
signals:

```text
useful profiles: 2/8
model time:      52.86s
API errors:       0
```

Visual inspection showed that an empty result included a sharp, full-frame
white/red/black motorcycle and rider. The failure was therefore the prompt,
not insufficient image detail. The instruction was narrowed to prohibit race-
number transcription while explicitly requiring every obvious color and
pattern.

The revised prompt produced:

```text
useful profiles: 8/8
model time:      38.02s
API errors:       0
```

Repeated `10` profiles consistently described a white motorcycle with red and
black accents. Repeated `957` profiles consistently emphasized orange/purple
motorcycle and rider colors, although the blurrier sighting contained fewer
attributes.

The two human-confirmed `69` crops visibly depict different motorcycles and
riders sharing the same number. Qwen described one as predominantly
black/white/red and the other as blue/white/red. This confirms that the event
registry must support multiple visual variants beneath one race-number string.

Conclusion:

- the revised number-blind metadata prompt is useful enough for a broader
  experiment;
- metadata calls remain expensive at about 4.75 seconds per crop and should be
  concentrated on golden variants and unresolved queries;
- metadata is supporting evidence and requires normalization plus human-
  validated matching tests before it can affect routing;
- race-number records must contain multiple variants rather than one canonical
  profile.

## 2026-09-04 — Incremental Refactor: Identifier Module

The first behavior-preserving extraction from the 3,012-line
`test_pipeline.py` moved race-number normalization into
`racesort/identifiers.py`. The pipeline now imports the shared helper instead
of defining it inline. Older experiment scripts retain their local copies so
their historical behavior is not silently changed.

Six focused unit tests verify:

- `0` is valid;
- `00` and `007` preserve leading zeros;
- numeric and alphanumeric identifiers remain strings;
- lowercase input is normalized to uppercase;
- missing, `UNKNOWN`, overlong, and invalid-character inputs are rejected.

Python compilation and the existing pipeline-output regression checker passed
with zero failures. The checker reported the two already-observed operational
warnings for Qwen call-count and review-workload variation. No inference,
threshold, routing, or output-schema behavior changed.

## 2026-09-04 — Incremental Refactor: Quality Module

The second behavior-preserving extraction moved the validated sharpness
measurement, conservative non-primary filter, and absolute blur filter into
`racesort/quality.py`. Threshold values remain explicit configuration in
`test_pipeline.py` and are passed into the shared decision functions.

Eight new quality tests cover:

- the multi-vehicle requirement for non-primary filtering;
- strict relative-area and relative-sharpness threshold boundaries;
- strict absolute-blur threshold behavior;
- edge-rich imagery scoring higher than a uniform image.

Together with the identifier tests, all 14 unit tests passed. Recomputing the
quality score for all 33 existing regression crops produced exact matches with
their stored pipeline results; the maximum absolute difference was `0.0`.
Python compilation and the existing output regression checker also passed with
zero failures and the same two known Qwen-variation warnings. No threshold,
routing, inference, or output-schema behavior changed.

## 2026-09-04 — Event-Day Operating Model Clarified

The photographer supplied the expected production cadence:

```text
groups:                         A, B, C
motorcycles per group:          approximately 30
session length:                 20 minutes
cycles per group:               5
total sessions per day:         15
shooting duration:              approximately 5 hours
photographs per day:            approximately 4,000
unique motorcycles normally:    75–100
unique motorcycles maximum:     150
```

The first complete A/B/C cycle is now the proposed human-confirmation
checkpoint. It should provide initial coverage of all three rider groups before
the remaining four cycles are processed with registry assistance.

Event, group, cycle, and session must remain separate context fields. Group can
narrow suggestions but cannot prove identity, particularly because multiple
motorcycles may share a race-number string. Performance planning must report a
4,000-photo event-day projection in addition to the existing 1,000-photo
projection.

The established 19-photo regression output remained valid after the identifier
and quality refactors:

```text
passed checks: 94
warnings:       2
failed checks:  0
```

The warnings were the known Qwen verification-call and human-review-workload
variation. No safety regression was observed.

## 2026-09-04 — Incremental Refactor: Detection Module

The third behavior-preserving extraction moved detector-box geometry and the
opt-in merged-motorcycle recovery policy into `racesort/detection.py`. The six
validated geometric thresholds remain grouped and visible in
`test_pipeline.py` through an immutable `MergedBoxCriteria` value.

Eight new tests cover:

- non-negative box area;
- intersection, containment, and IoU calculations;
- safe handling of zero-area boxes;
- unchanged baseline behavior while recovery is disabled;
- no split for non-motorcycle classes;
- replacement of one valid merged parent with two children;
- rejection of unbalanced child pairs;
- rejection of children without enough horizontal separation.

All 22 identifier, quality, and detection unit tests passed. Python compilation
and the existing output regression checker passed with zero failures and the
same two known Qwen-variation warnings. `test_pipeline.py` decreased from 2,934
to 2,782 lines without changing thresholds, inference, routing, or outputs.

## 2026-09-04 — Event-Day Clock and Sales Deadline

The production schedule was clarified:

```text
09:00  cycle 1: A, B, C
10:00  cycle 2: A, B, C
11:00  cycle 3: A, B, C
12:00  cycle 4: A, B, C
13:00–14:00 lunch
14:00  cycle 5: A, B, C
15:00–18:00 photograph sales
```

Each A/B/C cycle consists of three consecutive 20-minute sessions. The first
complete cycle ends around 10am and remains the proposed registry-confirmation
checkpoint, but later shooting starts immediately. The review checkpoint must
therefore be brief and must not prevent safe background detection, quality
analysis, or resumable processing.

When media is available throughout the day, the production target is useful,
sales-ready sorting by 3pm. Unresolved cases should be prioritized for human
attention before the sales window rather than presented as one undifferentiated
queue after all processing finishes.

## 2026-09-04 — Incremental Refactor: Configuration Module

The fourth behavior-preserving extraction moved pipeline settings into the
immutable `RaceSortConfig` class in `racesort/config.py`. `test_pipeline.py`
temporarily exposes compatibility aliases so its inference and routing body can
remain unchanged while the application is split into smaller modules.

The module preserves the current models and thresholds as defaults, validates
environment overrides, and adds optional event context fields for event ID,
date, group, cycle, and session. These fields remain separate from race-number
identifiers. Run summaries now include that event context and a 4,000-photo
event-day time projection alongside the existing 1,000-photo projection.

Twelve configuration tests cover defaults, string and path handling, boolean
parsing, event normalization, and rejection of invalid paths, dates, groups,
cycles, race types, thresholds, and merged-box geometry. All 34 unit tests
passed. Python compilation and the existing output regression checker passed
with 94 checks, the same two known Qwen-variation warnings, and zero failures.
`test_pipeline.py` decreased from 2,782 to 2,731 lines.

## 2026-09-04 — Incremental Refactor: Prompt Module

The fifth behavior-preserving extraction moved all Qwen prompt text and prompt
selection into `racesort/prompts.py`. This includes the direct number prompt,
the two historical fast-number prompts, constrained OCR-candidate verification,
and the motorcycle and car metadata schemas. The working pipeline now imports
the prompt policy instead of embedding or constructing it inline.

A syntax-tree comparison confirmed that all five static prompt strings match
the pre-refactor version exactly. Five focused tests protect leading zeros,
alphanumeric identifiers, OCR candidate constraints, vehicle-specific metadata
schemas, and rejection of unsupported race types. All 39 unit tests passed.
Python compilation and the existing regression checker passed with 94 checks,
the same two known Qwen-variation warnings, and zero failures.
