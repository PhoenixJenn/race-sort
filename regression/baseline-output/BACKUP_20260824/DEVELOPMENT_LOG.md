# RaceSort Development Log

## 2026-08-21 to 2026-08-23 --- Initial Feasibility

### Problem

Defined a local AI workflow for a motorsports photographer shooting
\~1,000 photos/hour for \~8 hours/day. Existing workflow requires an
assistant to inspect every photo and type visible race numbers;
multi-vehicle photos belong to multiple number folders.

### Hardware

-   MacBook Pro
-   M3 Pro
-   18 GB unified memory
-   macOS Sequoia 15.6
-   \~100 GB free disk

Decision: avoid large 27B-class models for routine local inference.

### Model

Selected Qwen3-VL because this is a vision problem.

Initial `qwen3-vl:4b` interaction was excessively verbose. Switched to:

`qwen3-vl:4b-instruct`

### Ollama

Version: `0.32.15`.

### Image-input troubleshooting

Interactive attempts that pasted prompts/image paths did not reliably
provide the image and caused long monologues. Lesson: explicitly attach
images as vision input; changing Terminal's working directory alone does
not grant model filesystem vision.

### Whole-image test: GGBM0021

Known obvious numbers: 49 and 706.

Result:

`49 70`

49 correct; 706 misread as 70.

Conclusion: whole-scene multi-number recognition is not sufficiently
reliable by itself.

### Full-resolution focused test

Result:

``` text
Error: 400 Bad Request
request (4113 tokens) exceeds the available context size (4096 tokens)
n_prompt_tokens: 4113
n_ctx: 4096
```

Conclusion: full-resolution photos consume too much visual context.

### Resize experiment

``` bash
sips -Z 1500 GGBM0021.JPG --out GGBM0021-small.jpg
```

Focused foreground-number prompt returned `706` correctly.

### GGBM0022 focused tests

After resize: - central motorcycle → `215` correct - left motorcycle →
`49` correct

Working hypothesis: detect/crop one vehicle first, then ask Qwen to read
that crop. Postpone dedicated OCR until testing shows it is needed.

------------------------------------------------------------------------

## 2026-08-23 --- Python Integration

### Environment

Python: `3.12.4`

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install ollama pillow
```

### Test folder

Current working folder name is `test-photos` (hyphen). Keep paths
consistent unless deliberately renamed.

### test_qwen.py

The script loads `test-photos/GGBM0021.JPG`, resizes with Pillow using
`thumbnail((1500, 1500))`, saves an analysis JPEG, calls local Ollama,
passes the image through the `images` field, and prints Qwen's result.

Successful output:

``` text
Created resized image: test-photos/GGBM0021-small.jpg
Qwen response:
706
```

### Milestone

**Phase 1 proof of concept complete.**

Proven: - local vision inference works on M3 Pro / 18 GB - Python
controls the workflow - image resizing works - Qwen correctly reads
tested focused race numbers

------------------------------------------------------------------------

## 2026-08-23 --- Phase 2A: Automatic Motorcycle Detection

### Detection dependencies

Installed:

``` bash
pip install torch torchvision transformers
```

Verified: - PyTorch: `2.13.0` - Transformers: `5.15.1`

### Missing timm dependency

First run of `test_detector.py` failed with:

``` text
ImportError:
TimmBackbone requires the timm library but it was not found in your environment.
```

Installed:

``` bash
pip install timm
```

Lesson: with the current Transformers/DETR setup, `timm` is an explicit
project dependency.

The error message suggested restarting the runtime after installation.
Because RaceSort is running as normal Terminal Python processes rather
than a persistent notebook kernel, no virtual-environment rebuild was
needed. A new Python invocation was sufficient.

### Hugging Face warning

The successful run displayed:

``` text
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

This is currently non-blocking. No HF token is required for the
prototype.

### DETR weight-load report

`facebook/detr-resnet-50` loaded with several `UNEXPECTED`
`num_batches_tracked` entries. Transformers notes these can be ignored
when loading across differing task/architecture details. The model
completed inference successfully, so no action is currently required.

### GGBM0021 detector result

Terminal output:

``` text
Motorcycle 1: confidence=0.99, box=(4071, 1079, 4892, 1843)
Motorcycle 2: confidence=1.00, box=(543, 988, 1434, 1888)
Motorcycle 3: confidence=0.98, box=(1586, 1120, 2915, 2659)
Motorcycles detected: 3

Saved debug image: test-photos/GGBM0021-detected.jpg
```

### Visual validation

Opened `GGBM0021-detected.jpg`.

All three DETR boxes correctly enclosed useful/visible motorcycles: -
left motorcycle with #49 - foreground motorcycle with #706 -
right/background motorcycle

The photograph actually contains approximately five motorcycles. Two
additional motorcycles are heavily obscured/background vehicles and were
not detected.

This is currently considered acceptable and potentially desirable
behavior because the two omitted motorcycles do not present useful
visible race numbers for sorting.

### New evaluation principle

Do not judge the detector solely on:

> Did it detect every motorcycle/car fragment in the image?

The more useful RaceSort metric is:

> Did it detect every **sortable vehicle**---a vehicle sufficiently
> visible to provide a usable race-number classification?

Future benchmarking should track sortable-vehicle recall separately from
raw object-detection recall.

### Milestone

**Phase 2A automatic motorcycle detection complete for the first
regression image.**

Proven: - DETR loads and runs locally - `facebook/detr-resnet-50`
identifies the three useful motorcycles in GGBM0021 - bounding boxes are
visually correct - detection confidence is high (0.98--1.00) - originals
remain unchanged; debug output is saved separately

### Next milestone --- Phase 2B

Use the DETR bounding boxes to crop each detected motorcycle and send
each crop independently to Qwen3-VL.

Target:

``` text
GGBM0021.JPG
Vehicle 1 → 49
Vehicle 2 → 706
Vehicle 3 → UNKNOWN
```

This will be the first end-to-end test where RaceSort finds the vehicles
and reads their numbers without manually describing which motorcycle
Qwen should inspect.

------------------------------------------------------------------------

## Decision Log

### Qwen3-VL instead of Qwen3.8

RaceSort is fundamentally visual.

### 4B Instruct

Fits 18 GB unified memory and avoids unwanted verbose thinking behavior.

### Resize before VLM inference

Full-resolution input exceeded the 4096-token context and is
unnecessarily expensive.

### 1500 px longest edge

Works on current samples. This remains an experimental value, not a
permanent production specification.

### DETR for initial vehicle detection

`facebook/detr-resnet-50` successfully detected the three useful
motorcycles in the first regression image. Continue testing before
treating it as the final detector.

### timm is required

The current DETR/Transformers environment requires `timm`; include it in
setup instructions.

### Sortable-vehicle recall matters more than raw count

Heavily obscured vehicles with no usable race-number evidence do not
need to be prioritized merely to maximize detection count.

### Postpone OCR

Focused Qwen recognition correctly returned 706, 215, and 49. Test the
simpler architecture first.

### Preserve originals

Analysis must never destructively alter production photographs.

### Human review stays in the design

The goal is drastic reduction in manual classification, not pretending
AI will be perfect.
