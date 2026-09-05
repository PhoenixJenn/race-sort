# RaceSort Product Requirements Document

## Document Status

- Product: RaceSort
- Status: Active development
- Product stage: Validated recognition pipeline; pre-production workflow
- Primary technical source of truth: `PROJECT_CONTEXT.md`
- Experiment history: `DEVELOPMENT_LOG.md`
- Developer onboarding: `START-HERE.md`

This document defines what RaceSort must accomplish as a product. If a technical threshold, architecture detail, or current implementation status conflicts with `PROJECT_CONTEXT.md`, `PROJECT_CONTEXT.md` takes precedence.

## 1. Product Summary

RaceSort is a fully local, AI-assisted workflow for motorsports event photographers. It analyzes event photographs, identifies visible race-number identifiers on motorcycles or cars, and associates each photograph with every applicable race number.

RaceSort is intended to reduce repetitive manual inspection and typing without pretending uncertain AI output is reliable. High-confidence assignments should be automated, unsuitable vehicle crops should be rejected cheaply, and unresolved cases should be presented for efficient human review.

Original photographs are permanent source assets and must never be destructively altered by RaceSort.

## 2. Problem Statement

A motorsports photographer may capture approximately:

- 4,000 photographs over approximately five shooting hours per event day;
- three groups of approximately 30 motorcycles each;
- five 20-minute cycles per group, producing 15 sessions per day;
- 75–100 unique motorcycles normally and up to 150 maximum;
- multiple useful vehicles in a single photograph.

The current manual workflow requires an assistant to inspect each image and enter every visible race number. This creates a large, time-sensitive sorting workload at events.

AI recognition can reduce that workload, but individual OCR and vision-model outputs are imperfect. RaceSort therefore needs an evidence-based routing system that automates defensible cases while preserving human control over uncertainty.

## 3. Product Goals

RaceSort must:

1. Dramatically reduce manual photograph inspection and race-number typing.
2. Automatically assign only sufficiently supported race numbers.
3. Support multiple useful vehicles and multiple assignments per photograph.
4. Remove clearly non-primary or unsellable crops before expensive inference.
5. Make uncertain cases fast for a human to resolve.
6. Preserve evidence and provenance for every automated decision.
7. Preserve original photographs unchanged.
8. Operate fully offline at a racetrack after installation and setup.
9. Use one hardware-agnostic codebase on Apple Silicon macOS and Windows/NVIDIA systems.
10. Remain understandable and maintainable by a developer who is learning Python.

## 4. Non-Goals

RaceSort is not intended to:

- guarantee that every visible vehicle is detected;
- force a race number when evidence is ambiguous;
- eliminate human review;
- treat AI output as ground truth merely because it looks plausible;
- modify, rename, move, overwrite, or delete original photographs during analysis;
- become separate Mac and Windows application forks;
- depend on internet access during event operation;
- optimize for tiny or heavily obscured background vehicles that cannot produce useful photographs;
- build a polished end-user UI before recognition and regression behavior are stable.

## 5. Users

### Primary User: Event Photographer

Needs to ingest a large event shoot, receive trustworthy race-number associations, review unresolved cases, and export organized deliverables quickly.

### Secondary User: Photography Assistant

Needs a fast review interface that minimizes typing and presents the vehicle crop, proposed number, confidence evidence, and decision history together.

### Development User

Needs incremental, testable Python changes; explicit setup and test commands; readable output; and protection against accidental regression or destructive file handling.

## 6. Core User Workflow

1. The user selects or configures an input directory of original photographs.
2. RaceSort performs a preflight check for local models, dependencies, storage, and writable output paths.
3. RaceSort reads originals without modifying them.
4. RaceSort detects sortable vehicles and creates separate analysis crops or proxies.
5. RaceSort filters conservative non-primary and too-blurry crops.
6. RaceSort routes remaining crops through OCR and vision-model recognition.
7. After the first complete A/B/C cycle, RaceSort presents proposed numbers and visual variants for human confirmation.
8. RaceSort builds an event-scoped, multi-variant registry from confirmed evidence.
9. RaceSort resolves later-session candidates using independently confirmed evidence.
10. RaceSort automatically assigns supported identifiers.
11. RaceSort presents unresolved cases for human review.
12. RaceSort exports or copies deliverables without altering source originals.

## 7. Current Recognition Architecture

```text
Original photograph
        ↓
analysis/proxy representation
        ↓
DETR vehicle detection
        ↓
individual vehicle crops
        ↓
non-primary filter
        ↓
absolute blur/sellability filter
        ↓
RapidOCR + unanchored direct Qwen read
        ↓
OCR/direct agreement?
   ┌────┴────┐
  yes        no
   ↓          ↓
anchored     QWEN_CANDIDATE / REVIEW
verification
   ↓
all three agree exactly?
   ┌────┴────┐
  yes        no
   ↓          ↓
CONFIRMED    QWEN_CANDIDATE / REVIEW
                  ↓
       registry + independent DINO
                  ↓
 CORROBORATED / KNOWN_NUMBER_REVIEW /
       CONFLICTING / UNSUPPORTED / REVIEW
```

The ordering of direct reading and anchored verification may be optimized when doing so does not weaken the three-signal confirmation requirement.

## 8. Functional Requirements

### FR-1: Original Asset Protection

- RaceSort must open original photographs read-only for analysis.
- RaceSort must not overwrite or destructively edit originals.
- RaceSort must not rename, move, or delete originals during recognition.
- Crops, proxies, caches, metadata, and output copies must be stored separately.
- Any future export operation must copy, link, or reference originals through an explicitly safe mechanism.

### FR-2: Vehicle Detection

- RaceSort must detect supported race-vehicle classes, initially motorcycles and cars.
- Detection must support more than one useful vehicle per photograph.
- Detected crops must retain a reference to their source photograph and detection evidence.
- Detection success must be evaluated using sortable-vehicle recall rather than recall of every background fragment.

### FR-3: Quality and Role Filtering

- RaceSort must distinguish vehicle role from photograph sellability.
- A crop may be `PRIMARY` or `SECONDARY` while also being `TOO_BLURRY`.
- Current conservative non-primary rule:

```text
vehicles_in_photo > 1
AND relative_area < 0.20
AND relative_sharpness < 0.45
→ FILTERED_NON_PRIMARY
```

- Current conservative blur rule:

```text
absolute sharpness < 150
→ FILTERED_TOO_BLURRY
```

- Filtered crops must not consume OCR or Qwen inference.
- Thresholds must remain configurable and regression-tested as the labeled dataset grows.

### FR-4: Race-Number Data Model

- Race numbers must be opaque strings, never integers.
- Leading zeros must be preserved.
- `"007"` and `"7"` must remain distinct.
- `"0"` must remain a valid identifier and must not be treated as missing data.
- Alphanumeric identifiers must not break processing.
- Current accepted normalized format is `[A-Z0-9]{1,6}`.
- Prompts must explicitly preserve leading zeros and visible character order.

### FR-5: OCR and Vision Routing

- RapidOCR must provide the inexpensive first evidence layer.
- Every crop reaching recognition must receive an unanchored direct Qwen number read under the current conservative policy.
- Candidate-anchored Qwen verification must not be treated as independent of its OCR suggestion.
- Automatic confirmation requires exact agreement among:
  - the OCR candidate;
  - candidate-anchored Qwen verification;
  - an unanchored direct Qwen read.
- If OCR and the direct read disagree, the direct result may remain `QWEN_CANDIDATE` but must not become `CONFIRMED`.
- A direct Qwen read by itself must never become `CONFIRMED`.
- Empty or ambiguous reads must route to human review rather than force a number.

### FR-6: Candidate Resolution

- Qwen candidates must be compared with known confirmed identities when independent evidence exists.
- A race number is a non-unique string label and must not be used as a unique motorcycle or rider key.
- An event registry must support multiple motorcycle/rider variants for the same race number.
- Each variant may store multiple confirmed viewpoint references and structured metadata.
- A mismatch with one variant must not reject a candidate until the other confirmed variants for that number have been considered.
- A candidate crop must never be used as its own DINO reference.
- Missing reference evidence must not be treated as conflicting evidence.
- Current resolution states are:
  - `CORROBORATED`;
  - `KNOWN_NUMBER_REVIEW`;
  - `CONFLICTING`;
  - `UNSUPPORTED`;
  - `REVIEW`.
- Current provisional DINO promotion rule:

```text
candidate matches a confirmed registry number
AND best independent DINO similarity >= 0.90
→ CORROBORATED
```

- The `0.90` threshold is provisional and must be evaluated against larger representative datasets.

### FR-7: Photo-Level Assignment

- One photograph may be associated with multiple race numbers.
- Multiple distinct motorcycles may share the same race number and must sort to the same race-number destination.
- Only `CONFIRMED` and `CORROBORATED` identifiers may enter automatic photo assignments.
- Candidate or review states must not silently enter confirmed assignments.
- Duplicate identifiers within a photograph must be collapsed without converting identifiers to numbers.

### FR-8: Human Review

- Human review must show the original-photo context or an appropriate proxy plus the applicable crop.
- Review must expose OCR, Qwen, DINO, registry, filtering, and provenance evidence where available.
- Human labels must keep these concepts separate:
  - actual race number;
  - number readability;
  - vehicle role;
  - image quality;
  - correctness or disposition;
  - notes.
- Review must minimize typing and make accepting, correcting, or rejecting a candidate efficient.

### FR-9: Evidence and Provenance

- Each vehicle result must retain its source photo and crop identity.
- Results must retain detector confidence, quality metrics, OCR candidates, raw Qwen responses, route, timings, DINO references, similarity, threshold, decision, and reasons where applicable.
- Machine-readable per-photo and batch summaries must be generated.
- Automated assignments must be explainable from preserved evidence.

### FR-10: Regression Validation

- The existing 19-photo / 33-crop dataset must remain the active regression suite until superseded deliberately.
- Stable safety invariants must fail the regression command when broken.
- Normal model variability that only increases conservative human review may produce a warning rather than a failure.
- The regression suite must cover at least:
  - the two known blurry primary crops;
  - OCR recovery of `54` and `721`;
  - race number `"0"`;
  - numberless-motorcycle hallucination containment;
  - multiple vehicles in one photograph;
  - string identifier validation;
  - DINO self-match prevention;
  - assignment safety.

### FR-11: Offline Operation

- RaceSort must perform event-time inference without internet access.
- Required models must be installed and cached before event operation.
- Production execution must not silently download missing models.
- A future preflight command must verify model and dependency availability and provide a clear actionable error when something is missing.

### FR-12: Hardware-Agnostic Operation

- RaceSort must remain one application and codebase.
- Hardware-specific choices must be configuration, not application forks.
- Planned profiles include:
  - `auto`;
  - `mac_apple_silicon`;
  - `windows_nvidia`;
  - `cpu`.
- Configurable settings must eventually include model names, device choices, worker counts, concurrency limits, and input/output paths.
- Recognition policy and safety thresholds must not silently change by platform.
- Optimizations must be validated independently of any one machine.

### FR-13: Performance Reporting

- Every benchmark run must report:
  - photographs processed;
  - vehicles detected;
  - filter counts;
  - OCR candidate and empty counts;
  - Qwen verification and direct-call counts;
  - final disposition counts;
  - total human-review workload;
  - DETR, OCR, Qwen, DINO, and total timing;
  - average seconds per photograph;
  - projected 1,000-photo duration.
- Production planning must also report projected 4,000-photo event-day duration.
- A machine-readable `run-summary.json` must preserve these metrics for comparison.
- Performance changes must be compared across repeated warm runs because local model startup and runtime state can cause large timing variation.
- Accuracy and assignment safety take priority over throughput.

## 9. Non-Functional Requirements

### Reliability

- A failed or missing evidence layer must degrade to review, not an invented assignment.
- Partial results should remain inspectable after an interruption where practical.
- Generated output must be reproducible enough to diagnose model variability.

### Maintainability

- Model-specific code should remain replaceable.
- Changes should be incremental and regression-tested.
- Large refactors must wait until working behavior is protected.
- Code changes should include exact test commands and expected outcomes.

### Portability

- Paths must use cross-platform abstractions such as `pathlib`.
- Application behavior must not depend on macOS-only shell commands or filesystem conventions.
- GPU acceleration must be optional, with a CPU fallback.

### Privacy

- Event photographs and recognition evidence must remain local unless the user explicitly exports them.
- RaceSort must not require a cloud inference service for event operation.

## 10. Current Validated Benchmark

Dataset:

```text
19 photographs
33 detected vehicle crops
```

Stable routing structure:

```text
FILTERED_NON_PRIMARY: 7
FILTERED_TOO_BLURRY: 2
OCR candidate cases: 13
OCR empty cases: 11
Qwen verification calls after direct-first routing: approximately 9–10
Qwen direct calls under current safety policy: 24
```

Recent correct runs have produced approximately:

```text
CONFIRMED: 9–10
CORROBORATED: 4
KNOWN_NUMBER_REVIEW: 2
UNSUPPORTED: 2–4
REVIEW: 4–6
human-review workload: 10–11
```

The disposition variation is caused by local vision-model variability. Conservative movement into review is acceptable; unsupported automatic assignment is not.

Observed serial timing varies substantially with local Qwen/Ollama runtime state:

```text
cold/slow observed run: approximately 165 seconds
warm observed runs: approximately 51–57 seconds
warm serial midpoint: approximately 54 seconds
```

These values are development baselines, not final product commitments.

## 11. Acceptance Criteria for a Production-Ready Recognition Core

The recognition core is ready to support production workflow development when:

1. Original-file protection is demonstrated and tested.
2. The regression checker reports no safety failures.
3. Known blurry, recovery, number-zero, hallucination, multi-vehicle, and self-match cases pass.
4. Automatic assignments contain only `CONFIRMED` or `CORROBORATED` identifiers.
5. Models load from local resources with the internet disabled.
6. A clean startup procedure succeeds from a new terminal session.
7. Hardware selection and concurrency are configuration-driven.
8. A representative larger event dataset has been human-labeled and evaluated.
9. False-confirmation and false-filter rates meet an explicitly approved operational threshold.
10. The review workflow demonstrates meaningful labor savings over manual sorting.

## 12. Milestones

### Milestone 1: Consolidated Regression Pipeline — Substantially Complete

- DETR, quality filters, OCR/Qwen routing, independent DINO, evidence output, performance summary, and regression checking are integrated.
- Remaining work includes continued accuracy validation and removal of obsolete experimental dependencies from the main script when safe.

### Milestone 2: Hardware-Agnostic Accuracy and Performance

- Preserve current safety behavior.
- Eliminate logically unnecessary model calls.
- Establish repeated cold and warm baselines.
- Test concurrency one stage at a time.
- Compare counts, decisions, memory behavior, and timing after every change.

### Milestone 3: Configuration and Reproducibility

- Add hardware profiles and explicit overrides.
- Add clean startup documentation.
- Add offline model/dependency preflight.
- Verify consistent behavior on supported platforms.

### Milestone 4: Event Workflow

- Implement safe ingest and event-state management.
- Record event, group, cycle, and session context separately from race numbers.
- Support a human-confirmation checkpoint after the first complete A/B/C cycle.
- Implement persistent identity registry behavior.
- Provide an efficient human-review workflow.
- Implement non-destructive export/copy organization.

### Milestone 5: Production Readiness

- Validate on larger representative events.
- Measure false confirmations, false filters, throughput, and review labor.
- Test recovery from interruptions and low-storage conditions.
- Complete macOS and Windows deployment documentation.

## 13. Key Product Metrics

RaceSort should track:

- photographs processed per minute;
- projected time per 1,000 photographs;
- sortable vehicles detected;
- automatically confirmed assignments;
- independently corroborated assignments;
- human-review cases;
- review cases per photograph;
- false confirmations;
- false filters;
- corrected candidates;
- average human review time per case;
- total labor saved versus the manual workflow;
- cold-start and warm-run processing time;
- peak CPU, system-memory, GPU, and VRAM usage where measurable.

False-confirmation and false-filter rates are more important than maximizing automatic-assignment percentage.

## 14. Risks and Mitigations

### Vision-Model Hallucination

Risk: Qwen can return plausible numbers for numberless or ambiguous motorcycles.

Mitigation: direct reads remain candidates; automatic confirmation requires independent agreement; regression tests protect known hallucination cases.

### Anchored Verification Bias

Risk: a Qwen verification prompt containing an OCR candidate can repeat an incorrect candidate.

Mitigation: require an unanchored direct read and exact three-signal agreement before confirmation.

### DINO Viewpoint Sensitivity

Risk: the same vehicle can produce lower similarity across viewpoints.

Mitigation: use DINO as supporting evidence, preserve review states, use multiple independent references when available, and keep the threshold provisional.

### Model Runtime Variability

Risk: identical runs can have substantially different Qwen timing and slightly different candidate/review outcomes.

Mitigation: compare repeated warm benchmarks, persist run summaries, keep conservative outputs in review, and separate safety failures from workload warnings.

### Small Regression Dataset

Risk: thresholds can appear reliable on 33 crops but fail at a larger event.

Mitigation: preserve current thresholds, expand human-labeled data deliberately, and measure false filters and false confirmations before changing policy.

### Hardware Constraints

Risk: a configuration that performs well on one machine may exhaust memory or perform poorly elsewhere.

Mitigation: maintain hardware-agnostic logic, configurable devices and concurrency, CPU fallback, and per-machine benchmarking.

## 15. Open Product Decisions

- What false-confirmation rate is acceptable for production automatic assignment?
- What false-filter rate is acceptable for non-primary and blur rejection?
- How should a persistent event registry be initialized, updated, corrected, and carried between event days?
- Should corroborated candidates be exported immediately or remain visible in a lightweight audit queue?
- What review-interface interaction produces the lowest time per unresolved crop?
- Which safe output mechanism should be the default: copies, filesystem links, manifests, or a combination?
- What photograph and vehicle classes must the first production release support beyond motorcycles and cars?
- How should model installation and offline preflight be packaged for non-technical users?

## 16. Release Guardrails

No release may:

- destructively modify original photographs;
- convert race-number identifiers to integers;
- treat `"0"` as missing;
- drop leading zeros;
- use a crop as its own DINO evidence;
- automatically assign an uncorroborated direct Qwen read;
- silently download models during event operation;
- hide uncertainty by forcing a race number;
- introduce platform-specific recognition policy forks;
- bypass the regression suite for a performance improvement.
