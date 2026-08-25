# RaceSort — START HERE

## Purpose of This File

This file is the handoff document for continuing RaceSort in a fresh ChatGPT session or local AI-assisted development environment without having to reconstruct the project from the original conversation.

**Start every new RaceSort development session by reading this file, then `PROJECT_CONTEXT.md`.**

The three project documents have different jobs:

1. **`START-HERE.md`** — onboarding, handoff instructions, current working position, and prompts for a fresh AI session.
2. **`PROJECT_CONTEXT.md`** — source of truth for the current architecture, requirements, validated rules, thresholds, hardware targets, and immediate milestone.
3. **`DEVELOPMENT_LOG.md`** — experiment history: what was tried, measured, changed, rejected, and learned.

If these documents disagree about current architecture, **`PROJECT_CONTEXT.md` wins**. Use `DEVELOPMENT_LOG.md` to understand why a decision was made, not to restore an older design.

---

# 1. What RaceSort Is

RaceSort is a **fully local AI-assisted motorsports photography workflow**.

The photographer may shoot approximately:

- 1,000 photographs/hour
- roughly 8 hours/day

The current manual workflow requires a human assistant to inspect each photograph, identify motorcycle/car race numbers, and associate the photo with the applicable vehicle number(s).

RaceSort's goal is to automate as much of that work as can be done safely while keeping uncertain cases easy for a human to review.

A photograph may contain multiple useful vehicles, so one original photograph may need to be associated with multiple race numbers.

RaceSort is not trying to achieve artificial “100% AI accuracy.” The operational objective is:

- automatically resolve high-confidence cases;
- cheaply discard unusable/non-primary vehicle crops before expensive inference;
- minimize false confirmations;
- reduce human typing and inspection;
- provide efficient human review for unresolved cases;
- preserve evidence explaining automated decisions;
- remain fully offline at a racetrack.

---

# 2. Non-Negotiable Safety / Data Rules

## NEVER destructively modify original photographs.

Original photographs are source assets.

RaceSort must not:

- overwrite originals;
- alter originals;
- destructively rename originals during analysis;
- move originals as part of recognition;
- delete originals.

Use:

- crops;
- proxies;
- cached representations;
- embeddings;
- metadata;
- output copies;
- references to originals.

Any eventual sorting/export operation should preserve the original source and use copies, links, or another explicitly safe mechanism.

---

# 3. Critical Race-Number Rule

Race numbers are **opaque string identifiers**, not numbers in the programming sense.

Examples:

```text
007
7
0
54A
A12
```

Rules:

- `"007"` must remain `"007"`.
- `"007"` and `"7"` are different identifiers.
- `"0"` is a valid race number.
- Never convert race-number identifiers to integers for storage/comparison.
- Current racers primarily use numeric identifiers.
- Letters are not currently common, but an alphanumeric identifier must not break RaceSort.
- Current normalization permits `[A-Z0-9]{1,6}`.

Prompts sent to vision models should explicitly say to preserve leading zeros and visible character order.

---

# 4. Development Philosophy

The developer is learning Python/software engineering while building RaceSort.

Therefore:

- work incrementally;
- explain unfamiliar Python concepts;
- do not dump large architectural rewrites without walking through them;
- give exact insertion/replacement locations when modifying code;
- prefer complete replacement blocks when that is safer;
- test each component before adding another;
- avoid unnecessary frameworks/dependencies;
- do not refactor working code merely for elegance;
- measure before optimizing;
- keep experiments isolated until validated.

When giving code changes, be explicit about:

1. which file to open;
2. what text/function to find;
3. whether to insert before/after or replace it;
4. the exact replacement code;
5. the command to run;
6. what successful output should roughly look like.

Do not assume advanced Python knowledge.

---

# 5. Current Development Hardware

## Development Mac

Current development/reference machine:

```text
MacBook Pro
Apple M3 Pro
18 GB unified memory
macOS Sequoia 15.6
Python 3.12.4
Ollama
```

## Photographer's Windows Target

The photographer's machine is:

```text
Intel Core i7-8700
6 cores / 12 threads
3.2 GHz base
up to 4.6 GHz turbo
48 GB RAM
NVIDIA GTX 1050 Ti
4 GB VRAM
```

The GTX 1050 Ti's **4 GB VRAM is an important constraint**.

RaceSort must eventually support both machines using **one codebase**.

Do not create a Mac fork and Windows fork.

The future architecture should use configuration/hardware profiles such as:

```text
auto
mac_apple_silicon
windows_nvidia
cpu
```

Potential configurable items include:

- PyTorch device;
- detector device;
- DINO device;
- Qwen/Ollama model;
- OCR settings;
- worker count;
- OCR concurrency;
- DINO concurrency;
- maximum simultaneous vision-model calls;
- input/output/review paths.

This configuration layer is planned but **is not the immediate next task**.

First consolidate and validate the current pipeline.

---

# 6. Offline Requirement

RaceSort must operate **without internet access at the racetrack**.

Current AI components are intended to run locally.

Production operation must not silently download models.

Eventually RaceSort needs a preflight/setup check that verifies required models and dependencies are available before an event.

---

# 7. Current Core Components

The project has experimentally validated:

```text
Python
Ollama
Qwen3-VL 4B Instruct
facebook/detr-resnet-50
RapidOCR
facebook/dinov2-small
Pillow
PyTorch
torchvision
Transformers
timm
```

Current responsibilities:

### DETR

Find useful motorcycles/vehicles in the photograph and create vehicle crops.

### Image-quality metrics

Measure things such as:

- crop area;
- relative area;
- sharpness;
- relative sharpness.

These metrics are used to avoid wasting expensive inference on non-primary or unusable crops.

### RapidOCR

Cheap first attempt at extracting race-number candidates.

Measured around ~0.2–0.3 seconds/crop in the current tests, making it much cheaper than Qwen.

### Qwen3-VL

Used for visual number verification/direct reading.

Qwen is powerful but:

- comparatively expensive;
- capable of hallucinating plausible race numbers.

Therefore direct Qwen output is **not automatically truth**.

### DINOv2

Creates visual vehicle embeddings.

Used as independent evidence that a crop visually resembles confirmed sightings of a known race-number identity.

DINO is extremely cheap compared with Qwen once loaded.

---

# 8. Current Validated Routing Logic

The current experimental architecture is approximately:

```text
ORIGINAL PHOTO
      ↓
DETR
      ↓
VEHICLE CROPS
      ↓
NON-PRIMARY FILTER
      ↓
BLUR / SELLABILITY FILTER
      ↓
RapidOCR
   ┌──┴───┐
candidate none
   ↓       ↓
Qwen      Qwen
verify    direct read
   ↓       ↓
   └── candidate/decision
             ↓
       REGISTRY + DINO
             ↓
       FINAL DISPOSITION
```

More explicitly:

```text
DETR crop
   │
   ├─ non-primary?
   │      └─ FILTERED_NON_PRIMARY
   │
   ├─ too blurry?
   │      └─ FILTERED_TOO_BLURRY
   │
   └─ process
          ↓
       RapidOCR
       /       \
candidate       none
   ↓             ↓
Qwen direct read
   ↓             ↓
matches OCR?    candidate?
 /      \        /       \
yes      no     yes       no
 ↓        ↓      ↓         ↓
Qwen     QWEN_  QWEN_    REVIEW
verifies CAND.  CANDIDATE
   ↓        \     /
agrees?      \   /
 /    \       \ /
yes    no candidate/review
 ↓      ↓        /
CONFIRMED QWEN_CAND./REVIEW
             \   /
              ↓
       candidate resolution
              ↓
 registry + independent DINO
```

---

# 9. Validated Non-Primary Filter

Current conservative rule:

```text
vehicles_in_photo > 1
AND relative_area < 0.20
AND relative_sharpness < 0.45
→ FILTERED_NON_PRIMARY
```

On the current 33-crop benchmark it filtered 7 crops.

Human review considered those filters accurate.

Important:

**NON_PRIMARY is not the same thing as TOO_BLURRY.**

A crop can be a primary motorcycle but still be too blurry to sell.

---

# 10. Validated Blur / Sellability Rule

Human review uses separate image-quality labels:

```text
SELLABLE
BORDERLINE
TOO_BLURRY
```

Current conservative automatic blur rule:

```text
absolute sharpness < 150
→ FILTERED_TOO_BLURRY
```

Two important examples caught by this rule:

```text
GGBM0005 motorcycle-01
sharpness ≈ 121.7
human = TOO_BLURRY

GGBM0006 motorcycle-01
sharpness ≈ 108.9
human = TOO_BLURRY
```

Before this gate, Qwen hallucinated race numbers on these blurry crops.

The blur filter now prevents them from reaching expensive Qwen inference.

---

# 11. OCR + Qwen Rules

## OCR candidate + anchored Qwen agreement

OCR plus candidate-list verification is not enough for automatic confirmation. The anchored verification prompt once repeated the incorrect OCR value `122` for a motorcycle whose visible number was `721`.

Current conservative policy:

```text
OCR candidate
+ Qwen verifies that exact candidate
+ unanchored direct Qwen read returns the same identifier
→ CONFIRMED
```

If the direct read disagrees, keep the direct result as `QWEN_CANDIDATE`. If it is unreadable, use `REVIEW`.

## OCR rejected by Qwen

Do not immediately give up.

A direct Qwen fallback recovered cases including:

```text
OCR C42A → Qwen direct 54
OCR 122  → Qwen direct 721
```

## Direct Qwen read

A direct Qwen read must **NOT** automatically become `CONFIRMED`.

Use:

```text
QWEN_CANDIDATE
```

Reason: Qwen has hallucinated plausible race numbers on clear motorcycles that actually have no visible number.

Numberless Aprilia crops were important examples.

---

# 12. Candidate Resolution States

Current states:

```text
CORROBORATED
KNOWN_NUMBER_REVIEW
CONFLICTING
UNSUPPORTED
```

## CORROBORATED

Independent evidence strongly supports the Qwen candidate.

## KNOWN_NUMBER_REVIEW

Qwen read a race number that already exists in the registry, but insufficient independent evidence exists for automatic promotion.

This is **not a failure**.

Missing evidence is not negative evidence.

## CONFLICTING

Another evidence source actively supports a different identity or conflicts with the Qwen candidate.

## UNSUPPORTED

The candidate is not known in the registry and lacks meaningful corroboration.

---

# 13. DINO Identity Rules

DINOv2 is supporting evidence, not an infallible identity system.

It is viewpoint-sensitive.

## Critical rule: NEVER use a crop as its own reference.

A self-comparison produces similarity near:

```text
1.0000
```

but provides zero independent identity evidence.

The DINO candidate-coverage experiment originally contained this bug and was corrected.

Any future implementation must explicitly exclude the candidate crop itself from reference comparisons.

## Missing reference != disagreement

If #98 has only one known crop, that crop cannot corroborate itself.

That should produce:

```text
KNOWN_NUMBER_REVIEW
```

rather than a rejection.

---

# 14. Current Experimental DINO Promotion Rule

Corrected independent DINO testing produced strong evidence for known Qwen candidates.

Observed best independent similarities included:

```text
#52  → 0.9654
#721 → 0.9149
#869 → 0.9387
#52  → 0.9534
#54  → 0.9537
#721 → 0.9149
```

#98 had no independent reference available.

Current provisional rule:

```text
Qwen candidate matches an existing registry number
AND
best INDEPENDENT DINO similarity >= 0.90
→ eligible for CORROBORATED
```

This threshold is **provisional** because the current regression dataset is small.

Do not silently treat 0.90 as a universal truth.

Continue regression testing it as the dataset grows.

---

# 15. Current Regression Dataset

The active benchmark contains:

```text
19 photographs
33 detected vehicle crops
```

This dataset is now important enough to treat as a regression suite rather than disposable test data.

It includes known examples covering:

- clear race numbers;
- multiple motorcycles;
- blurry motorcycles;
- background/non-primary motorcycles;
- bad OCR candidates;
- Qwen recovery cases;
- Qwen hallucination cases;
- known DINO identities;
- numberless Aprilia examples.

Preserve the current test photos and known-good outputs.

Do not casually delete `test-output/` until a regression snapshot/checkpoint exists.

---

# 16. Important Performance Results

## Original rich-Qwen pipeline

```text
19 photos
33 vehicles

Total batch time: 313.08s
Average seconds/photo: 16.48
Projected 1,000-photo time: 274.6 minutes
```

Qwen rich profiling dominated runtime.

## RapidOCR alone

```text
33 vehicles
OCR total: 7.17s
Average OCR: 0.217s/vehicle
```

This established OCR as a worthwhile cheap routing stage.

## OCR + conditional Qwen

```text
33 vehicles
Qwen calls: 13
Confirmed: 10
Review: 23
Batch elapsed: 89.28s
```

## Routing Pipeline V2

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
Average total / processed vehicle: 5.97s
```

Qwen remains the dominant cost.

## Corrected DINO candidate coverage

```text
Known Qwen candidates tested: 7
Embeddings computed: 12
Batch elapsed: 0.54s
```

DINO corroboration is extremely cheap compared with Qwen.

---

# 17. Human Review

A dynamic verbose HTML reviewer was built during development.

The reviewer should expose evidence alongside the vehicle crop rather than forcing the human to inspect CSV files manually.

Human validation concepts should remain separate:

```text
actual race number
number readability
vehicle role
image quality
correctness / disposition
notes
```

Vehicle role:

```text
PRIMARY
SECONDARY
NON_PRIMARY
```

Image quality:

```text
SELLABLE
BORDERLINE
TOO_BLURRY
```

Do not collapse these into one field.

Example:

```text
PRIMARY + TOO_BLURRY
```

is valid.

---

# 18. Existing Experimental Files

The project accumulated many test scripts.

Examples include:

```text
test_qwen.py
test_detector.py
test_fast_numbers.py
test_ocr.py
test_ocr_qwen.py
test_primary_filter.py
test_garbage_filter.py
test_routing_pipeline.py
test_routing_pipeline_v2.py
test_candidate_resolution.py
test_dino_candidate_coverage.py
```

Other useful scripts developed during earlier phases include:

```text
test_pipeline.py
build_registry.py
match_review.py
evaluate_candidates.py
build_photo_assignments.py
```

Do **not** assume every experimental script should become production code.

The experiments exist to document validated behavior.

The next phase should consolidate the successful ideas into a smaller maintainable application.

---

# 19. Directory Cleanup Guidance

Do not reorganize all Python files at once.

Many current scripts use relative paths such as:

```python
Path("test-output")
```

Moving them can accidentally break path resolution.

For now:

- preserve working Python locations until the consolidated pipeline runs;
- preserve `test-photos/`;
- preserve `test-output/`;
- HTML reviewers can safely be organized into a `reviewers/` folder;
- create `experiments/` for archival scripts after consolidation;
- create `checkpoints/` for known-good project snapshots.

Long-term target organization may resemble:

```text
race-sort/
│
├── START-HERE.md
├── PROJECT_CONTEXT.md
├── DEVELOPMENT_LOG.md
├── README.md
├── requirements.txt
│
├── src/
│   └── racesort/
│       ├── pipeline.py
│       ├── detection.py
│       ├── quality.py
│       ├── ocr.py
│       ├── vision.py
│       ├── embeddings.py
│       ├── registry.py
│       └── config.py
│
├── scripts/
├── experiments/
├── reviewers/
├── regression/
├── working/
└── checkpoints/
```

Do not force this structure immediately. Migrate incrementally after the consolidated pipeline works.

---

# 20. Current Exact Project Position

**The exploratory component-testing phase is substantially complete.**

Do not immediately invent another isolated AI experiment.

The next milestone recorded in `PROJECT_CONTEXT.md` is:

> Consolidate the validated routing behavior into the main RaceSort pipeline.

The next end-to-end run should use the same 19-photo / 33-crop regression set.

It should report:

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

total human-review workload

DETR time
OCR time
Qwen time
DINO time
total elapsed time

average seconds/photo
projected 1,000-photo processing time
```

Correctness comes before parallelism.

---

# 21. What Comes After Consolidation

Once the consolidated pipeline reproduces or improves the validated behavior:

## Step 1 — Regression validation

Confirm the known cases still behave correctly.

Especially inspect:

- blurry GGBM0005 / GGBM0006 cases;
- #54 recovery;
- #721 recovery;
- numberless Aprilia hallucination cases;
- leading-zero behavior;
- race number `0`;
- multi-bike photographs.

## Step 2 — Parallelism

Benchmark safe concurrency.

Good initial candidates:

```text
preprocessing
quality metrics
RapidOCR
DINO embeddings
file I/O
```

Do not automatically parallelize Qwen.

Benchmark it separately.

The GTX 1050 Ti's 4 GB VRAM may require only one vision-model worker.

## Step 3 — Configuration Layer

Create the cross-platform configuration/hardware abstraction.

Goal:

```text
one RaceSort codebase
+
hardware/config profile
```

not separate Mac/Windows apps.

## Step 4 — Clean Startup Test

Close Terminal/Ollama/session state and prove RaceSort can be started from scratch with documented commands.

## Step 5 — Windows Deployment

Install and test on the photographer's i7-8700 / 48 GB / GTX 1050 Ti machine.

## Step 6 — Offline Preflight

Verify the full application works with internet disabled.

## Step 7 — Production Workflow/UI

Only after the recognition pipeline is stable should substantial effort go into:

- production photo ingest;
- output folder/copy behavior;
- persistent registry/event state;
- polished reviewer;
- photographer-facing UI.

---

# 22. Instructions for a Fresh AI Development Session

When transferring this project to another ChatGPT session or local AI coding environment, give the AI access to the RaceSort project directory.

Then use the following prompt.

## Recommended Initial Prompt

Copy/paste this:

```text
We are continuing an existing project called RaceSort.

Before making any code changes:

1. Read START-HERE.md.
2. Read PROJECT_CONTEXT.md completely. Treat PROJECT_CONTEXT.md as the source of truth for the current architecture, requirements, thresholds, and project state.
3. Read DEVELOPMENT_LOG.md as experiment history. Do not restore obsolete approaches merely because they appear in the log.
4. Inspect the current project directory and tell me what files/folders actually exist.
5. Inspect the current working pipeline and the most relevant recent experimental scripts, especially test_pipeline.py, test_routing_pipeline_v2.py, test_candidate_resolution.py, and test_dino_candidate_coverage.py if they exist.
6. Do not modify anything yet.

Then give me:
- a concise summary of the current RaceSort architecture;
- which files appear to be current working code versus experiments;
- any discrepancies you see between the documentation and actual code;
- the smallest safe plan for the next milestone.

Important working rules:
- Preserve original photographs. Never design destructive operations on originals.
- Race numbers are string identifiers. Preserve leading zeros such as 007. A race number of 0 is valid. Alphanumeric identifiers must not break the system.
- Work incrementally because I am learning Python.
- When we make code changes, tell me exactly which file, what section to find, what to replace/insert, and how to test it.
- Do not perform a large refactor before the current regression pipeline is working.
- RaceSort must ultimately run fully offline and support both Apple Silicon macOS and Windows/NVIDIA through configuration rather than separate application forks.

Our immediate milestone should be the one documented in PROJECT_CONTEXT.md unless the actual files reveal a blocking problem.
```

---

# 23. Prompt to Start the Consolidation Work

After the new AI session has inspected the project and its assessment looks correct, use:

```text
Let's start the consolidation milestone.

Do not rewrite the entire application at once.

First identify the current main processing loop and show me which validated pieces from the experimental scripts need to be integrated into it.

I want to proceed one component at a time and run a regression test after each meaningful change.

The final consolidated benchmark must use the existing 19-photo / 33-crop regression dataset and report the metrics listed in PROJECT_CONTEXT.md.

Start with the smallest safe first change. Explain exactly where it goes because I am still learning Python.
```

---

# 24. Prompt if the New Session Wants to Refactor Too Much

Use this if necessary:

```text
Do not refactor the project for cleanliness yet.

The priority is preserving the experimentally validated behavior and getting the consolidated pipeline working against the existing regression dataset.

Use the current code where practical. Make the smallest change necessary, test it, and then continue.

We can reorganize modules and directories after the consolidated pipeline passes regression testing.
```

---

# 25. Prompt for Debugging

If something breaks:

```text
Help me debug this incrementally.

Do not give me multiple speculative fixes at once.

First inspect the exact error and the relevant code. Explain in plain language what failed and why.

Then give me one change to make, tell me exactly where to make it, and tell me what command to run afterward.

Do not change unrelated working parts of RaceSort.
```

---

# 26. Prompt for a Clean Restart Test

When ready to verify reproducibility:

```text
I want to prove RaceSort can start from a completely fresh local session.

Assume I have closed Terminal and stopped any local model services.

Using the actual files in this project, give me the exact startup procedure from opening Terminal through activating the Python environment, starting/verifying required local services, and running the regression pipeline.

Do not guess commands if the project files can tell you the correct ones.

Afterward, help me turn the verified procedure into project documentation.
```

---

# 27. Prompt for Performance / Parallelism Phase

Use only after the consolidated pipeline is correct:

```text
The consolidated RaceSort pipeline now passes our regression test.

Let's benchmark parallelism without changing recognition behavior.

Start by identifying which stages can safely execute concurrently and which share constrained model/GPU resources.

Test one concurrency change at a time and compare:
- total elapsed time;
- per-stage timing;
- accuracy/dispositions;
- memory behavior;
- Qwen call count.

Do not assume more workers are faster.

Remember that the eventual Windows target has an i7-8700, 48 GB RAM, and GTX 1050 Ti with only 4 GB VRAM.
```

---

# 28. Prompt for Cross-Platform Configuration Phase

Use after performance behavior is understood:

```text
Let's implement the RaceSort hardware/configuration layer.

Requirements:
- one codebase for Mac and Windows;
- fully offline operation;
- auto-detection where practical;
- explicit overrides;
- Apple Silicon/MPS support where appropriate;
- Windows/NVIDIA/CUDA support where appropriate;
- CPU fallback;
- configurable Qwen model/backend and concurrency;
- configurable paths;
- no recognition-policy differences hidden inside platform-specific code.

The photographer's Windows reference machine is:
Intel i7-8700, 6 cores / 12 threads, 48 GB RAM, GTX 1050 Ti with 4 GB VRAM.

Proceed incrementally and preserve the regression behavior.
```

---

# 29. Questions the New AI Should NOT Re-Litigate Without New Evidence

The following have already been experimentally established and documented.

Do not casually restart these debates:

### “Should we try OCR?”

Yes. RapidOCR was measured and is cheap enough to be useful.

### “Should direct Qwen reads be automatically trusted?”

No. Qwen hallucinated plausible numbers on numberless motorcycles.

### “Can we treat race numbers as integers?”

No. `007` must be preserved, and `0` is valid.

### “Can DINO compare a candidate to itself?”

No. Self-match is invalid corroboration.

### “Does no DINO reference mean the candidate is wrong?”

No. Missing evidence is not negative evidence.

### “Should blurry/non-primary crops go through expensive Qwen processing?”

Not when the validated conservative filters safely remove them.

### “Should we make separate Mac and Windows RaceSort apps?”

No. One configurable codebase.

### “Should we reorganize everything before the pipeline works?”

No. Consolidate and regression-test first.

---

# 30. Important Caveat About Current Thresholds

The following are **validated on the current small regression dataset**, not universal constants:

```text
relative_area < 0.20
relative_sharpness < 0.45
absolute sharpness < 150
independent DINO best similarity >= 0.90
```

Do not “optimize” them casually.

Also do not assume they are permanently correct.

As more human-labeled race photographs become available, evaluate false-positive/false-negative behavior and revise thresholds based on evidence.

---

# 31. How to Use the Three Documentation Files

At the beginning of a new development session:

```text
START-HERE.md
      ↓
PROJECT_CONTEXT.md
      ↓
actual source code
      ↓
DEVELOPMENT_LOG.md when historical context is needed
```

Do not make the AI read hundreds of old chat messages as its primary project memory.

The repository documentation should carry the project forward.

When an important architectural decision changes:

1. update `PROJECT_CONTEXT.md`;
2. append the experiment/result to `DEVELOPMENT_LOG.md`;
3. update `START-HERE.md` only if the handoff/startup instructions or major project phase changes.

This keeps future sessions reproducible.

---

# 32. Immediate Handoff Checklist

Before switching to a new local development session, verify the RaceSort folder contains:

```text
START-HERE.md
PROJECT_CONTEXT.md
DEVELOPMENT_LOG.md
```

Also preserve:

```text
test-photos/
test-output/
```

and the current Python scripts, especially the latest pipeline/routing/candidate/DINO experiments.

Do not delete this original ChatGPT conversation yet. Keep it as historical backup until the new local session has:

1. read the documentation;
2. inspected the actual project;
3. correctly summarized the architecture;
4. identified the immediate milestone;
5. successfully run or begun working with the regression pipeline.

Once those five things happen, the project has effectively been handed off.

---

# 33. Current Handoff Point

The correct next sentence for a fresh RaceSort development session is:

> **We have finished the exploratory component experiments. Now consolidate the validated routing, filtering, OCR/Qwen, and independent-DINO candidate-resolution behavior into the main RaceSort pipeline, incrementally, and validate it against the existing 19-photo / 33-crop regression dataset before optimizing or reorganizing the application.**

That is where development should resume.
