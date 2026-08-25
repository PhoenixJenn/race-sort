# RaceSort

RaceSort is a local AI-assisted workflow for a motorsports event photographer. Its primary goal is to identify visible race numbers on motorcycles and race cars automatically, organize photographs by vehicle number, and reduce manual review while preserving the originals.

## Problem

The photographer may shoot about 1,000 photos/hour for roughly 8 hours/day. The current workflow displays one photo at a time while an assistant manually types each visible vehicle number. The existing application creates/uses a folder for that number and associates the photograph with it. A photo containing multiple numbered vehicles belongs to every applicable number.

## Target Workflow

1. Import race photographs.
2. Preserve originals unchanged.
3. Detect useful/visible cars or motorcycles.
4. Crop each detected vehicle.
5. Run a local vision-language-model profile pass.
6. Run an independent number-verification pass.
7. Accept only validated numbers; otherwise route the vehicle to REVIEW.
8. Save structured per-photo results.
9. Build a persistent vehicle registry from confirmed sightings.
10. Use second-pass visual identity matching to recover REVIEW observations.
11. Associate one original photo with multiple confirmed/inferred race numbers when appropriate.
12. Export/copy full original photographs into number-based folders.

## Current Environment

- Apple MacBook Pro, M3 Pro
- 18 GB unified memory
- macOS Sequoia 15.6
- ~100 GB free disk at project start
- Python 3.12.4
- Ollama 0.32.15
- Vision model: `qwen3-vl:4b-instruct`
- Vehicle detector: `facebook/detr-resnet-50`
- PyTorch 2.13.0
- Transformers 5.15.1
- Additional Python packages:
  - `ollama`
  - `Pillow`
  - `torch`
  - `torchvision`
  - `transformers`
  - `timm`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ollama pillow torch torchvision transformers timm
ollama pull qwen3-vl:4b-instruct
```

## Completed Milestones

### Phase 1 — Local vision recognition

Focused resized Qwen3-VL tests correctly read `706`, `215`, and `49`.

### Phase 2A — Automatic motorcycle detection

DETR successfully detected the useful/visible motorcycles in the original regression image. This established the metric of **sortable-vehicle recall**, rather than raw detection of every vehicle fragment.

### Phase 2B — End-to-end detection, crop, and number recognition

The first complete pipeline autonomously performed:

```text
photo → DETR → vehicle crops → Qwen → photo number aggregation
```

For `GGBM0021.JPG`, it produced:

```text
UNKNOWN
49
706
```

and correctly associated the original photo with `[49, 706]`.

### Phase 2B validation improvements

Single-pass Qwen output was not reliable enough. Observed failure modes included:

- blank-plate hallucinations
- stylized-font misreads
- blurry-number disagreements

The pipeline was upgraded to:

1. JSON vehicle profile pass
2. deterministic validation
3. independent number-verification pass
4. confirmation only when evidence agrees

Qwen now provides observations; RaceSort makes the classification decision.

### Phase 2C — Structured results and vehicle registry

The batch pipeline now writes one `photo-results.json` per photograph.

`build_registry.py` consolidates confirmed sightings into `vehicle-registry.json` while retaining REVIEW observations.

Current registry checkpoint contains 8 confirmed identities:

- 49
- 52
- 54
- 98
- 721
- 866
- 869
- 999

There are 15 confirmed sightings total plus retained REVIEW observations.

## Current Architecture

```text
Original Photograph
        ↓
DETR Detection
        ↓
Vehicle Crop
        ↓
Qwen JSON Profile Pass
        ↓
Deterministic Validation
        ↓
Independent Number Verification
        ↓
CONFIRMED or REVIEW
        ↓
Per-photo JSON
        ↓
Vehicle Registry
        ↓
Phase 2D: Visual Identity Matching
```

## Next Milestone — Phase 2D

Add visual embeddings so RaceSort can compare REVIEW crops against confirmed vehicle crops.

Initial goals:

1. Pick one lightweight embedding model suitable for the M3 Pro / 18 GB Mac.
2. Compute one embedding vector per vehicle crop.
3. Compare known same-bike and different-bike pairs.
4. Test especially blurry #54 REVIEW crops, repeated blank-Aprilia crops, and clearly different bikes.
5. Do not auto-assign identities yet.
6. Establish useful similarity behavior before choosing thresholds.

## Rules

- Never modify or overwrite original photographs.
- Never accept a race number from a single unvalidated Qwen read.
- Ambiguous observations go to REVIEW.
- Distinguish READ numbers from later INFERRED identities.
- Preserve provenance for every classification.
