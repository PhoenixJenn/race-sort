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
