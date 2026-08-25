# RaceSort Development Log

## 2026-08-21 to 2026-08-23 — Initial Feasibility

Defined RaceSort for a motorsports photographer shooting roughly 1,000 photos/hour for about 8 hours/day.

Hardware:

- MacBook Pro M3 Pro
- 18 GB unified memory
- macOS Sequoia 15.6

Selected `qwen3-vl:4b-instruct` via Ollama 0.32.15.

### Early findings

- Interactive image paths were unreliable; images must be attached explicitly.
- Whole-scene Qwen reading returned `49 70` for a photo containing 49 and 706.
- Full-resolution focused input exceeded the 4096-token context.
- Resizing and focusing produced correct reads of 706, 215, and 49.

---

## Phase 1 — Python Integration

Python 3.12.4.

Installed:

```bash
pip install ollama pillow
```

`test_qwen.py` successfully performed:

```text
Python → Pillow resize → Ollama → Qwen3-VL → result
```

Observed:

```text
Qwen response:
706
```

Phase 1 complete.

---

## Phase 2A — Automatic Vehicle Detection

Installed:

```bash
pip install torch torchvision transformers
```

Verified:

- PyTorch 2.13.0
- Transformers 5.15.1

DETR initially failed because `timm` was missing.

Resolved with:

```bash
pip install timm
```

Model:

`facebook/detr-resnet-50`

On GGBM0021 DETR detected the 3 useful motorcycles with confidence approximately 0.98–1.00.

This established **sortable-vehicle recall** as the relevant metric.

Phase 2A complete.

---

## Phase 2B — End-to-End Pipeline

Created `test_pipeline.py`.

Initial GGBM0021 result:

```text
UNKNOWN
49
706
```

The original photo could therefore be associated with 49 and 706.

### 19-photo batch

Observed key Qwen failure modes:

- blank-plate hallucinations
- blurry #54 disagreements
- stylized #866 misread

### Validation architecture added

The pipeline was changed to use:

1. JSON vehicle profile pass
2. deterministic validation
3. independent number-verification pass
4. agreement-based confirmation

Qwen now produces observations; RaceSort decides.

### Regression outcomes

- Blank Aprilia hallucinations no longer became confirmed tags.
- Blurry #54 disagreements remained REVIEW.
- #52 confirmed correctly.
- #866 produced 866 + 866 and was CONFIRMED.

Phase 2B complete.

---

## Performance Observation

The first 19-photo batch took approximately 3 minutes.

Future optimization candidates:

- time DETR and Qwen separately
- use Apple MPS for DETR if available
- avoid unnecessary second Qwen calls
- reduce crop size if accuracy permits
- test OCR as a fast first reader
- reserve Qwen for ambiguous cases
- exploit cross-photo identity

Performance optimization remains deferred until architecture/accuracy are more stable.

---

## Phase 2C — Structured Results

Added one `photo-results.json` per photo.

A placement bug initially caused structured results to be written incorrectly when the code block was outside the per-photo loop. The placement was corrected.

`photo-results.json` is now the machine-readable source of truth for registry building.

---

## Phase 2C — Vehicle Registry

Created `build_registry.py`.

It reads all per-photo JSON files and builds:

`test-output/vehicle-registry.json`

Current confirmed identities:

- 49 — 1 sighting
- 52 — 3 sightings
- 54 — 1 sighting
- 98 — 1 sighting
- 721 — 3 sightings
- 866 — 1 sighting
- 869 — 3 sightings
- 999 — 2 sightings

Total:

- 8 confirmed identities
- 15 confirmed sightings
- REVIEW observations retained

### Useful registry examples

#52:
- BMW
- black
- white/black rider leathers

#869:
- BMW
- black/white/red

#721:
- orange/black
- yellow number plate
- Aprilia observed in at least one sighting

#54:
- black/orange/blue/red
- black/orange leathers
- REVIEW observations appear visually similar

Blank Aprilia:
- multiple REVIEW observations
- visually similar profile across frames
- inconsistent hallucinated numbers reinforce distrust of first-pass OCR

### Phase 2C conclusion

RaceSort now has:

```text
✓ vehicle detection
✓ crops
✓ JSON profiles
✓ independent verification
✓ deterministic validation
✓ per-photo JSON
✓ batch CSV
✓ persistent vehicle registry
✓ retained REVIEW observations
```

Phase 2C complete.

---

## Phase 2D Planned

Next: visual identity matching.

Goal:

```text
confirmed crop
   ↓
visual embedding
   ↓
similarity comparison
   ↑
REVIEW crop
```

Initial test cases:

- blurry #54 REVIEW crops vs confirmed #54
- repeated blank-Aprilia REVIEW crops against each other
- clearly different bikes such as #52 vs #869
- repeated known same-bike sightings such as #52 or #869

Rules:

- lightweight local embedding model
- no vector database yet
- inspect similarity distributions first
- do not auto-assign REVIEW vehicles until thresholds are empirically justified
- preserve READ vs INFERRED provenance
