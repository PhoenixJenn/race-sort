# RaceSort

RaceSort is a fully local, AI-assisted workflow for sorting motorsports event
photographs by visible race-number identifiers. It detects useful motorcycles
or cars, rejects conservative non-primary and unsellable crops, combines OCR,
Qwen, and DINO evidence, and sends uncertain cases to human review.

RaceSort never treats original photographs as disposable. Analysis uses crops,
metadata, caches, and references; originals must not be overwritten, renamed,
moved, or deleted by recognition processing.

## Current Status

The project has a working end-to-end regression pipeline in `test_pipeline.py`.
It is being split into small tested modules incrementally, without changing the
validated routing behavior.

```text
Unit tests: 54 passed
Regression: 94 passed, 2 known variability warnings, 0 failures
Dataset:    19 photographs, 33 detected vehicle crops
```

The warnings track expected local Qwen variability in verification-call count
and human-review workload. Conservative movement into review is not an unsafe
automatic assignment.

## Recognition Flow

```text
original photograph
  → DETR vehicle detection and crops
  → non-primary and blur/sellability filters
  → RapidOCR candidate extraction
  → unanchored direct Qwen number read
  → anchored verification when applicable
  → CONFIRMED, QWEN_CANDIDATE, or REVIEW
  → independent registry/DINO resolution
  → CORROBORATED, KNOWN_NUMBER_REVIEW, UNSUPPORTED, or REVIEW
```

Direct Qwen output is evidence, not truth. Automatic confirmation requires the
validated agreement policy described in `PROJECT_CONTEXT.md`.

## Race-Number Rules

Race numbers are strings, not integers:

- `007` and `7` are different identifiers.
- `0` is valid and must not be treated as missing.
- Alphanumeric identifiers such as `54A` and `A12` must remain safe.
- Current normalized format is `[A-Z0-9]{1,6}`.
- OCR candidates must contain at least one digit under the current conservative
  policy, which filters likely sponsor/logo text.

## Project Map

```text
START-HERE.md                   onboarding and handoff instructions
PROJECT_CONTEXT.md              technical source of truth and next milestone
RaceSortPRD.md                  product requirements and release guardrails
DEVELOPMENT_LOG.md              experiment and implementation history
test_pipeline.py                working end-to-end regression pipeline
racesort/config.py              settings and event context
racesort/identifiers.py         race-number string normalization
racesort/ocr.py                 RapidOCR candidate normalization/filtering
racesort/quality.py             blur and non-primary filters
racesort/detection.py           box geometry and merged-box recovery
racesort/prompts.py             number and metadata prompt policy
racesort/qwen.py                Ollama/Qwen wrapper and response cache
racesort/visual_matching.py     DINO device, embedding, and similarity helpers
tests/                          model-free unit tests
regression/                     stored-output regression checker and fixtures
reviewers/                      local human-review HTML tools
experiments/                    isolated benchmarks; not production entry points
```

Generated outputs, caches, model environments, and photographs are intentionally
excluded from Git. Do not commit API keys, photographs, or large model/output
files.

## Development Environment

Reference development system:

- Apple M3 Pro MacBook Pro with 18 GB unified memory
- Python 3.12.4
- Ollama with `qwen3-vl:4b-instruct`
- `facebook/detr-resnet-50`
- RapidOCR
- `facebook/dinov2-small`

The eventual Windows reference is an Intel i7-8700, 48 GB RAM, and NVIDIA GTX
1050 Ti with 4 GB VRAM. RaceSort must remain one configurable codebase with
CUDA, Apple MPS, and CPU fallbacks—not separate application forks.

## Start a Development Session

```bash
cd /path/to/race-sort
source .venv/bin/activate
python --version
python -c "import ollama, torch, transformers; print('Environment OK')"
ollama --version
```

Start Ollama separately if it is not already running:

```bash
ollama serve
```

The unit tests and stored-output regression check do not require a fresh Qwen
inference run:

```bash
python -m unittest discover -s tests -v
python regression/check_pipeline_results.py
```

Run the full pipeline only when new inference output is needed:

```bash
python test_pipeline.py
```

Settings are defined by `racesort/config.py`. Defaults target `test-photos/`
and `test-output/`.

## Documentation Practice

Update documentation with each milestone:

- `PROJECT_CONTEXT.md` when architecture, thresholds, state, or the next
  milestone changes;
- `RaceSortPRD.md` when product behavior, requirements, acceptance criteria, or
  release guardrails change;
- `README.md` when setup, commands, project structure, or current status changes;
- `DEVELOPMENT_LOG.md` with measured results and why a decision was made.

The immediate next structural milestone is extracting routing decisions into a
model-free, thoroughly tested policy module. After that, RaceSort can proceed
toward first-cycle human confirmation and the event-scoped multi-variant
registry required when different motorcycles share the same race number.
