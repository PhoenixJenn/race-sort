# RaceSort Project Context

## Purpose

RaceSort is a local AI-assisted workflow for a motorsports event photographer. It is intended to identify race numbers on motorcycles and cars, associate each photograph with the applicable vehicle numbers, and drastically reduce the amount of manual sorting required at race events.

A typical event day produces roughly 4,000 photographs over about five hours
of shooting. The existing workflow requires an assistant to inspect
photographs and manually type visible race numbers. A photograph containing
multiple useful numbered vehicles must be associated with every applicable
number.

RaceSort is not intended to make originals disposable or to eliminate human judgment. The goal is to automate high-confidence work, cheaply discard unusable/non-primary crops, and route uncertain cases into an efficient human-review workflow.

## Event-Day Operating Model

A normal motorcycle event day has:

- three rider groups: A, B, and C;
- approximately 30 motorcycles per group;
- 20-minute group sessions;
- five cycles through all three groups;
- 15 total sessions over approximately five hours;
- roughly 4,000 photographs;
- normally 75–100 unique motorcycles, with up to 150 maximum.

The first complete A/B/C cycle is the natural first-pass checkpoint because it
is the earliest point likely to expose the full day's participant population.
RaceSort should present a checkpoint after that cycle for a human to confirm
proposed numbers and visual variants without pausing safe background work. The
remaining four cycles can then use the confirmed event registry for cheaper
ranking and review suggestions.

Typical event-day schedule:

```text
09:00–10:00  cycle 1: A, B, C (20 minutes each)
10:00–11:00  cycle 2: A, B, C
11:00–12:00  cycle 3: A, B, C
12:00–13:00  cycle 4: A, B, C
13:00–14:00  lunch
14:00–15:00  cycle 5: A, B, C
15:00–18:00  photograph sales
```

The cycle-one confirmation checkpoint must be short and should run while later
shooting continues. It must not stop safe detection, quality analysis, or other
background work. When photographs are available during the day, RaceSort should
process incrementally and target sales-ready assignments by 15:00. Remaining
uncertain cases should be prioritized for human attention before sales begin.

Event, group, cycle, and session are contextual evidence and must be stored
separately from the race-number string. Group membership may narrow candidates
but must not be treated as identity proof or encoded into the race number.

## Non-Negotiable Requirements

- Preserve original photographs unchanged.
- Never destructively modify, rename, overwrite, or move an original as part of analysis.
- Analysis must use copies, proxies, crops, cached data, or in-memory transformations.
- Support multiple useful vehicles in one photograph.
- A single photograph may be associated with multiple race numbers.
- Support UNKNOWN / REVIEW rather than forcing a number.
- Preserve evidence and provenance for automated decisions.
- RaceSort must work fully offline at the racetrack once models/dependencies are installed.
- RaceSort should remain one codebase for macOS and Windows; hardware-specific behavior must be configuration-driven rather than implemented as separate application forks.
- Human review remains part of the production design.

## Success Criteria

Success is not “AI identifies every vehicle perfectly.”

The important operational goals are:

1. Reduce manual inspection and typing dramatically.
2. Automatically resolve high-confidence race-number assignments.
3. Avoid spending expensive model time on obviously non-primary or unsellable crops.
4. Make remaining REVIEW cases fast for a human to resolve.
5. Maintain very low false-confirmation and false-filter rates.
6. Eventually approach or beat the current manual workflow throughput.

For detection, the important metric is **sortable-vehicle recall**, not raw object-detection recall. Tiny, obscured, blurry, or background motorcycles that cannot produce a useful sellable photograph do not need to be treated as primary sorting targets.

## Race Number Data Model

Race numbers are opaque string identifiers, not integers.

Rules:

- Preserve leading zeros.
- `"007"` and `"7"` are different race numbers.
- Never convert a race number to an integer for storage or comparison.
- Current racers primarily use numeric identifiers.
- Future alphanumeric identifiers must not break the system.
- Examples that must remain valid: `"007"`, `"54A"`, `"A12"`.
- Current normalization accepts `[A-Z0-9]{1,6}`.
- Prompts must explicitly preserve visible leading zeros and character order.

## Current Reference Hardware

### Development Mac

- MacBook Pro
- Apple M3 Pro
- 18 GB unified memory
- macOS Sequoia 15.6
- Python 3.12.4
- Ollama used for local Qwen inference

### Photographer Windows Reference Target

- Intel Core i7-8700
- 6 cores / 12 threads
- 3.2 GHz base, up to 4.6 GHz turbo
- 48 GB RAM
- NVIDIA GTX 1050 Ti
- 4 GB VRAM

The Windows machine's 4 GB VRAM is a significant constraint. Do not assume the exact Mac inference configuration will be optimal on this machine.

Future configuration should support automatic or explicit platform profiles such as:

- `auto`
- `mac_apple_silicon`
- `windows_nvidia`
- CPU fallback

Hardware-dependent settings should eventually include model choice, PyTorch device, concurrency, worker counts, and other performance parameters.

## Current Core Technologies

Current/validated components include:

- Python
- Ollama
- Qwen3-VL 4B Instruct for local vision-language inference
- `facebook/detr-resnet-50` for vehicle detection
- RapidOCR as the cheap first-pass number-reading stage
- `facebook/dinov2-small` for visual vehicle embeddings
- Pillow
- PyTorch
- torchvision
- Transformers
- timm

`timm` is required by the current DETR/Transformers environment.

Keep AI components replaceable. Do not tightly couple application logic to a single model/runtime.

## Current Validated Routing Architecture

The current experimental architecture is:

```text
Original photograph
        ↓
analysis/proxy image
        ↓
DETR vehicle detection
        ↓
individual vehicle crops
        ↓
non-primary filter
        ↓
absolute blur/sellability filter
        ↓
RapidOCR
        ↓
   ┌────┴─────┐
candidate    none
   ↓           ↓
Qwen direct read
   ↓           ↓
matches OCR?  candidate?
 /     \       /       \
yes     no    yes       no
 ↓       ↓     ↓         ↓
Qwen    QWEN_ QWEN_    REVIEW
verify  CAND. CANDIDATE
   ↓      \     /
agrees?    \   /
 /    \     \ /
yes    no candidate/review
 ↓      ↓       /
CONF. QWEN_CAND./
       REVIEW
          \   /
       evidence resolution
                  ↓
       registry + independent DINO
                  ↓
 CORROBORATED / KNOWN_NUMBER_REVIEW /
       CONFLICTING / REVIEW
```

This routing architecture is now consolidated in `test_pipeline.py` and checked by `regression/check_pipeline_results.py` on the 19-photo / 33-crop regression batch.

## Validated Filtering Rules

### Non-Primary Filter

Current validated conservative rule:

```text
vehicles_in_photo > 1
AND relative_area < 0.20
AND relative_sharpness < 0.45
→ FILTERED_NON_PRIMARY
```

On the current 33-crop benchmark this filtered 7 crops, and human review judged those filters accurate.

Do not interpret this as a universal final threshold. Preserve it as the currently validated rule and continue regression testing as the dataset grows.

### Absolute Blur / Sellability Filter

Role and sellability are separate concepts.

A crop may be a PRIMARY or SECONDARY subject but still be too blurry to sell.

Human labels use:

- `SELLABLE`
- `BORDERLINE`
- `TOO_BLURRY`

Current conservative blur rule:

```text
absolute sharpness < 150
→ FILTERED_TOO_BLURRY
```

On the current benchmark this caught the two problematic blurry primary crops that had caused Qwen hallucinations:

- GGBM0005 motorcycle-01
- GGBM0006 motorcycle-01

The blur gate prevents wasting OCR/Qwen work on these unsellable crops.

Do not merge `TOO_BLURRY` with `NON_PRIMARY`; they describe different failure modes.

## OCR / Qwen Rules

RapidOCR is now a useful cheap first stage. Earlier versions of the project postponed OCR, but measured experiments demonstrated that OCR is fast enough to justify routing through it.

Important rules:

### OCR Candidate + Anchored Qwen Agreement

OCR plus candidate-list verification is not sufficient for automatic confirmation. The verification prompt is anchored by the OCR suggestion and can repeat a plausible but incorrect OCR value.

A regression run demonstrated this failure on GGBM0018 motorcycle-01:

```text
OCR candidate: 122
anchored Qwen verification: 122
actual visible number: 721
```

Current conservative confirmation rule:

```text
OCR candidate
+ anchored Qwen verification of that exact candidate
+ unanchored direct Qwen read of that exact identifier
→ CONFIRMED
```

If the direct read disagrees, its result remains `QWEN_CANDIDATE`. If it is unreadable, route the crop to `REVIEW`.

### OCR Candidate Rejected by Qwen

Do not immediately give up.

If Qwen rejects the OCR candidate, perform a direct number read. This recovered cases where OCR was badly wrong, including:

- OCR `C42A` → direct Qwen `54`
- OCR `122` → direct Qwen `721`

### Direct Qwen Reads Are Not Confirmations

A direct Qwen number read by itself is evidence, not truth.

Use:

```text
QWEN_CANDIDATE
```

not `CONFIRMED`.

This rule is required because Qwen has produced plausible-looking hallucinations on clear motorcycles with no visible race number, including numberless Aprilia cases.

A direct read must be corroborated before automatic assignment.

## Candidate Resolution

Current experimental candidate states:

- `CORROBORATED`
- `KNOWN_NUMBER_REVIEW`
- `CONFLICTING`
- `UNSUPPORTED`

Interpretation:

### CORROBORATED

Independent evidence strongly supports the Qwen candidate.

### KNOWN_NUMBER_REVIEW

The Qwen candidate is already a known registry number, but insufficient independent evidence is available for automatic promotion.

Missing evidence is not negative evidence.

### CONFLICTING

Another evidence layer actively supports a different identity or otherwise conflicts with the Qwen candidate.

### UNSUPPORTED

The candidate is not known in the registry and lacks meaningful independent corroboration.

## DINOv2 Identity Evidence

DINOv2 Small is used for local visual embeddings and vehicle similarity.

Important lessons:

- DINO is useful but viewpoint-sensitive.
- Similarity should not be treated as a universal identity oracle.
- Ranking, multiple sightings, metadata, and other evidence can matter.
- DINO is particularly useful for checking whether a Qwen-read known number visually resembles independently confirmed sightings of that number.

### Critical Self-Match Rule

**Never use a crop as its own DINO reference.**

A self-comparison produces similarity approximately 1.0 but provides zero independent identity evidence.

Candidate-resolution code must explicitly exclude the candidate crop from its reference set.

### Missing DINO Evidence

No independent DINO reference does **not** mean the candidate is wrong.

Example: a number with only one confirmed crop cannot use that same crop as its own corroborating reference. It should remain `KNOWN_NUMBER_REVIEW` until independent evidence exists.

### Current Experimental Promotion Threshold

On the current known-candidate coverage experiment, six independently comparable Qwen candidates had best independent DINO similarity above 0.90.

Current conservative experimental rule:

```text
Qwen candidate matches an existing registry number
AND best independent DINO similarity >= 0.90
→ eligible for CORROBORATED
```

This is an experimental threshold derived from the current small benchmark. It must be regression-tested on a larger representative dataset before becoming a permanent production rule.

## DINO Candidate Coverage Findings

After removing accidental self-comparisons, current known Qwen candidates showed:

- #52 → strong independent support
- #721 → strong independent support
- #869 → strong independent support
- #52 → strong independent support
- #54 → strong independent support
- #721 → strong independent support
- #98 → no independent reference available

The lack of independent evidence for #98 is not evidence against #98.

## Human Review

A dynamic verbose HTML reviewer exists for inspecting crops alongside automated evidence.

Human ground-truth concepts are intentionally separate:

- actual race number
- number readability
- vehicle role: PRIMARY / SECONDARY / NON_PRIMARY
- image quality: SELLABLE / BORDERLINE / TOO_BLURRY
- correctness/review status
- notes

This separation is important for training/calibrating rules. For example, a motorcycle may be PRIMARY but TOO_BLURRY.

Human labels should be treated as regression ground truth for future experiments.

## Registry and Evidence Fusion

Race numbers are non-unique string labels, not unique vehicle or rider IDs.
Within one event, more than one motorcycle/rider combination may legitimately
use the same race number. The registry must therefore map an event-scoped race
number to one or more visual variants, and each variant may contain multiple
confirmed viewpoint references and structured vehicle/rider metadata. All
variants still sort to the same race-number destination.

Do not collapse a race number to one canonical motorcycle, rider, embedding,
or profile. A visual mismatch with one known variant is not evidence that the
number is wrong when another variant for that same number may exist.

The project has experimented with:

- `vehicle-registry.json`
- `review-matches.json`
- `candidate-evaluations.json`
- photo-level assignment data

Earlier evidence fusion included:

- profile number
- verification number
- DINO rank
- DINO best similarity
- DINO mean similarity
- make compatibility
- vehicle colors
- rider leathers
- helmet colors

Candidate classes used in that phase:

- `STRONG_CANDIDATE`
- `POSSIBLE_CANDIDATE`
- `CONFLICT`
- `INSUFFICIENT_EVIDENCE`

Preserve this history. The newer routing/candidate-resolution work should reuse useful existing evidence rather than discard it.

## Performance Baselines

### Original Rich Qwen Pipeline

19 photos / 33 vehicles:

```text
Total batch time: 313.08s
DETR total: 10.55s
Qwen profile total: 280.37s
Qwen verification total: 16.58s
Average seconds/photo: 16.48
Projected 1,000-photo time: 274.6 minutes
Average profile call: 8.50s
Average verification call: 0.50s
```

Conclusion: rich Qwen profiling dominated runtime.

### Fast Two-Pass Number Experiment

33 vehicles:

```text
Confirmed: 14
Review: 19
Pass A total: 10.31s
Pass B total: 31.94s
Total elapsed: 42.26s
Average Pass A: 0.31s
Average Pass B: 0.97s
Average two-pass time/vehicle: 1.28s
```

### Integrated Fast/Rich Batch

19 photos / 33 vehicles:

```text
Fast-confirmed vehicles: 15
Vehicles for REVIEW: 18
Rich profiles generated: 27
Total batch time: 254.26s
DETR total: 11.81s
Fast Pass B total: 124.03s
Fast Pass A total: 10.64s
Rich profile total: 101.35s
Average fast two-pass: 4.08s
Average vehicle total: 7.17s
Average seconds/photo: 13.38
Projected 1,000-photo time: 223.0 minutes
```

### RapidOCR Baseline

33 vehicles:

```text
OCR total: 7.17s
Average OCR time/vehicle: 0.217s
```

This demonstrated that OCR is cheap enough to use as a routing stage.

### OCR + Conditional Qwen

33 vehicles:

```text
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

### Routing Pipeline V2

33 vehicle crops:

```text
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

Interpretation:

- Cheap filters successfully removed 9/33 crops before OCR/Qwen.
- The blur filter prevented known hallucination-prone unsellable crops from reaching Qwen.
- Direct Qwen fallback recovered useful candidates missed by OCR.
- Qwen remains the dominant runtime cost.
- Candidate corroboration should reuse registry/DINO evidence rather than add unnecessary Qwen calls.

### DINO Candidate Coverage

Corrected run:

```text
Known Qwen candidates tested: 7
Embeddings computed: 12
Batch elapsed: 0.54s
```

DINO corroboration is extremely cheap compared with Qwen once the model is loaded and appropriate reference crops exist.

## Current Benchmark Dataset

The active regression batch contains:

- 19 photos
- 33 detected vehicle crops

Preserve the current `test-output/` data as a regression dataset/checkpoint before major pipeline changes.

Earlier regression images include:

### GGBM0021.JPG

Known useful vehicles include:

- #706 prominent foreground motorcycle
- #49 left motorcycle
- another useful/visible motorcycle without a readable number

DETR successfully detected the useful vehicles while omitting heavily obscured/background motorcycles.

### GGBM0022.JPG

Known visible numbers include:

- #215 center
- #49 left

These early images remain useful sanity checks even though the larger 19-photo dataset is now the primary benchmark.

## Offline Operation

RaceSort must work without internet connectivity at the track.

Current model categories run locally:

- Qwen through Ollama
- DETR
- DINOv2
- RapidOCR

Production startup should not silently depend on downloading models.

Hugging Face models should be cached during setup and loaded in offline/local-only mode for event operation.

A future installation/preflight workflow should verify that all required models are present before the photographer leaves for an event.

## Cross-Platform Configuration Requirement

Do not fork RaceSort into separate Mac and Windows applications.

The intended design is:

```text
RaceSort application logic
        ↓
configuration / hardware profile
        ↓
device + model + concurrency selection
```

Future configuration should externalize at least:

- platform profile
- detector device
- embedding device
- vision backend/model
- OCR configuration
- worker counts
- parallel OCR behavior
- parallel embedding behavior
- maximum simultaneous vision-model calls
- input/output/review paths

Use `auto` where practical, with explicit overrides available.

Potential device resolution:

```text
Apple Silicon → MPS where supported
Windows + NVIDIA → CUDA where supported
otherwise → CPU fallback
```

Do not implement this abstraction until the current pipeline is consolidated and benchmarked. Preserve the requirement now so new code does not unnecessarily hard-code macOS assumptions.

## Concurrency / Parallelism

Parallelism is a planned optimization, not yet a validated production feature.

Likely candidates for safe parallel work:

- image preprocessing
- sharpness/quality metrics
- RapidOCR
- DINO embeddings
- file I/O

Qwen/Ollama concurrency must be benchmarked separately because memory pressure and GPU/VRAM limitations may make serialized or low-concurrency inference faster and safer, especially on the GTX 1050 Ti 4 GB target.

Do not assume “more threads = faster.”

## Engineering Principles

- Work incrementally.
- Get each component working and measured before adding the next.
- Prefer evidence-driven changes over speculative complexity.
- Keep experiments isolated until validated.
- Preserve regression outputs.
- Keep AI/model components replaceable.
- Treat missing evidence differently from conflicting evidence.
- Never use self-comparisons as identity corroboration.
- Never silently invent unreadable numbers.
- Never let a direct Qwen read become truth without appropriate validation.
- Preserve leading zeros in race numbers.
- Preserve originals.
- Optimize for operational human-labor reduction, not benchmark vanity.
- Keep RaceSort fully local/offline at event time.
- Maintain one cross-platform codebase.

## Current Project State

Validated experimentally:

- local Qwen vision inference
- DETR vehicle detection
- multi-vehicle crop processing
- race-number string normalization
- RapidOCR first-pass routing
- Qwen candidate verification
- direct-Qwen fallback
- non-primary filtering
- absolute blur/sellability filtering
- DINO visual embeddings
- registry matching
- evidence fusion
- candidate-resolution states
- dynamic human-review page
- timing/performance instrumentation
- consolidated end-to-end routing in `test_pipeline.py`
- validated environment-based pipeline configuration
- cross-platform DINO device selection: CUDA, Apple MPS, then CPU
- isolated prompt policy, Qwen client/cache, quality, identifier, detection,
  and DINO visual-matching modules
- code-only Git/GitHub workflow with photographs and generated outputs ignored

Not yet finalized:

- final automatic candidate-promotion policy
- larger-scale accuracy benchmark
- parallel processing strategy
- complete hardware profiles and Windows-specific performance tuning
- Windows deployment
- production export/copy workflow
- polished application UI

Current code organization:

```text
test_pipeline.py                 working end-to-end regression pipeline
racesort/config.py               validated settings and event context
racesort/identifiers.py          race-number string normalization
racesort/quality.py              blur and non-primary filters
racesort/detection.py            box geometry and merged-box recovery
racesort/prompts.py              number and metadata prompt policy
racesort/qwen.py                 Ollama/Qwen wrapper and response cache
racesort/visual_matching.py      DINO device, embedding, and similarity helpers
tests/                           model-free unit tests for extracted modules
regression/check_pipeline_results.py
                                 established output regression checker
```

As of 2026-09-04, all 49 unit tests pass. The established 19-photo / 33-crop
regression reports 94 passed checks, two known nondeterministic Qwen/workload
warnings, and zero failures. `test_pipeline.py` is 2,299 lines, down from 2,934
before the incremental extractions. The latest pushed code commit is `4b13aa9`
(`Extract DINO visual matching`), and local `master` matches `origin/master`.

## Immediate Next Milestone

Continue the incremental consolidation by extracting OCR candidate normalization
and filtering from `test_pipeline.py` into a small model-free module. Preserve
candidate order, leading zeros, valid `0`, alphanumeric safety, the requirement
that a candidate contain at least one digit, and exact existing routing
behavior. Add focused unit tests, run the complete unit suite and existing
regression checker, then commit the change locally before proceeding.

After OCR extraction:

1. Extract routing decisions into a thoroughly tested policy module.
2. Run a fresh full regression pipeline when an actual inference run is
   warranted, rather than relying only on stored-output checks.
3. Implement the first-cycle human confirmation and multi-variant event
   registry, allowing distinct motorcycles to share one race number.
4. Resume accuracy/performance evaluation on the larger labeled dataset.
5. Complete hardware profiles, clean-start/offline checks, and Windows testing.
