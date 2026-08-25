# RaceSort Project Context

## Purpose

RaceSort replaces a highly manual motorsports photography sorting workflow. A race-track photographer shoots motorcycles and cars whose participants apply vinyl race numbers. Volume can reach roughly 1,000 photographs/hour for about 8 hours/day.

## Primary Success Criterion

Success is not "AI identifies every photograph perfectly."

Success is:

> The assistant no longer manually inspects and types race numbers for thousands of photographs. High-confidence cases are automatic and uncertain cases are presented for fast review.

A wrong confident number is more damaging than an UNKNOWN. RaceSort should therefore bias toward precision and abstention.

## Detection Metric

Use **sortable-vehicle recall**:

> Did RaceSort detect every vehicle sufficiently visible to contribute a useful race-number classification?

Heavily obscured vehicles with no usable race-number evidence are lower priority.

## Current Hardware / Software

- Apple MacBook Pro, M3 Pro
- 18 GB unified memory
- macOS Sequoia 15.6
- Python 3.12.4
- Ollama 0.32.15
- PyTorch 2.13.0
- Transformers 5.15.1
- `timm`
- Pillow
- `qwen3-vl:4b-instruct`
- `facebook/detr-resnet-50`

## Current Processing Architecture

```text
Original Photograph
        ↓
DETR Object Detection
        ↓
Individual Vehicle Crops
        ↓
Qwen JSON Profile Pass
        ↓
Deterministic Profile Validation
        ↓
Independent Qwen Number Verification
        ↓
         ┌───────────────┐
         │               │
    CONFIRMED          REVIEW
         │               │
         └───────┬───────┘
                 ↓
        Per-photo structured JSON
                 ↓
          Vehicle Registry
                 ↓
       Phase 2D Visual Matching
```

## Important Findings

### Qwen image input

Images must be explicitly supplied as vision input.

### Verbosity

Use `qwen3-vl:4b-instruct`, not the thinking-oriented `qwen3-vl:4b`.

### Resize requirement

A full-resolution focused request exceeded the 4096-token context. Smaller focused crops are more reliable.

### Whole-scene versus focused recognition

Whole-scene multi-number reading is less reliable than single-vehicle crops.

## Phase 2B Validation Architecture

Observed single-pass failures included:

- blank-number Aprilia hallucinating different numbers in different frames
- blurry #54 producing inconsistent reads
- stylized #866 being misread

Current confirmation rule:

A number is CONFIRMED only when:
- the profile candidate survives deterministic validation
- the independent verifier returns a number
- both numbers agree

Otherwise the vehicle goes to REVIEW.

The profile is evidence, not ground truth.

## Per-photo Structured Results

Each processed photo now has `photo-results.json`, containing:

- photo name
- race type
- vehicle count
- confirmed photo numbers
- every vehicle result
- DETR confidence
- profile candidate
- verification candidate
- final number
- decision
- crop filename
- profile filename

## Phase 2C Vehicle Registry

`build_registry.py` creates:

`test-output/vehicle-registry.json`

The registry contains confirmed identities keyed by race number plus retained REVIEW observations.

### Current confirmed identities

- #49 — 1 sighting, Yamaha, blue/black
- #52 — 3 sightings, BMW, black, white/black leathers
- #54 — 1 sighting, black/orange/blue/red
- #98 — 1 sighting, black/green
- #721 — 3 sightings, orange/black, yellow plate, Aprilia observed
- #866 — 1 sighting, green/black, yellow plate
- #869 — 3 sightings, BMW, black/white/red
- #999 — 2 sightings, white/black

Total: 8 confirmed identities and 15 confirmed sightings.

## Useful REVIEW Patterns

### Blurry #54

Confirmed #54 profile and REVIEW observations have highly similar metadata. This is an ideal Phase 2D recovery case.

### Blank-number Aprilia

Multiple REVIEW observations appear to describe the same distinctive Aprilia, while Qwen number guesses vary. This is an ideal same-vehicle clustering case.

## Current Evaluation Philosophy

Failure severity:

1. Wrong confirmed number — highest severity
2. Missed readable number — lower severity
3. UNKNOWN / REVIEW on ambiguous evidence — acceptable

Never weaken the first-pass safety rule merely to improve recall. Recover ambiguous cases using second-pass cross-photo evidence.

## Phase Status

- Phase 1: local Qwen vision proof of concept — COMPLETE
- Phase 2A: vehicle detection — COMPLETE
- Phase 2B: detect → crop → profile → verify → validate — COMPLETE
- Phase 2C: structured results + vehicle registry — COMPLETE
- Phase 2D: visual identity matching — NEXT

## Phase 2D Objective

Use visual embeddings to compare actual vehicle crop images.

Initial work should:

1. Use a lightweight local embedding model.
2. Avoid vector databases initially.
3. Compute embeddings locally.
4. Compare known same-bike and different-bike pairs.
5. Measure similarity before choosing thresholds.
6. Use metadata as a supporting signal, not the sole matcher.
7. Do not auto-assign REVIEW observations until behavior is understood.


## Race Number Data Model

Race numbers are identifiers, not quantities.

RaceSort must store, compare, serialize, and display race numbers as strings
throughout the entire system.

Examples:

- `"007"` must remain `"007"` and must not become `"7"`.
- `"007"` and `"7"` represent different vehicle identities.
- `"54A"` and `"A12"` should remain valid identifiers if alphanumeric race
  numbers are encountered in the future.
- JSON registry keys and number fields must use strings.
- Folder names derived from race numbers must preserve the original identifier.

No component may normalize race numbers by converting them to integers.

## Engineering Principles

- Work incrementally.
- Preserve originals.
- Preserve regression cases.
- Keep AI components replaceable.
- Measure accuracy and speed.
- Preserve classification provenance.
- Distinguish READ numbers from later INFERRED identities.

- Treat race numbers as opaque string identifiers, never as numeric values.
  Preserve them exactly as observed, including leading zeros. For example,
  `"007"` and `"7"` are different race numbers.
- Race-number handling should be forward-compatible with alphanumeric
  identifiers such as `"54A"` or `"A12"`, even though current events use
  digits only.
- Never convert a race number to an integer or other numeric type anywhere
  in the pipeline, registry, JSON output, matching logic, folder naming,
  or future UI/database.
