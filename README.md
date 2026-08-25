# RaceSort

RaceSort is a local AI-assisted workflow for a motorsports event
photographer. Its primary goal is to identify visible race numbers on
motorcycles and race cars automatically, then organize photographs by
vehicle number while preserving the originals.

## Problem

The photographer may shoot about 1,000 photos/hour for roughly 8
hours/day. The current workflow displays one photo at a time while an
assistant manually types each visible vehicle number. The existing
application creates/uses a folder for that number and associates the
photograph with it. A photo containing multiple numbered vehicles
belongs to every applicable number.

## Target Workflow

1.  Import race photographs.
2.  Preserve originals unchanged.
3.  Create smaller analysis images/proxies.
4.  Detect useful/visible cars and motorcycles.
5.  Crop individual vehicles.
6.  Read visible race numbers locally.
7.  Assign confidence and send ambiguous cases to human review.
8.  Associate one photo with multiple numbers when necessary.
9.  Export/copy photos into number-based folders.
10. Later, optionally use visual similarity and metadata to improve
    identification.

## Current Environment

-   Apple MacBook Pro, M3 Pro
-   18 GB unified memory
-   macOS Sequoia 15.6
-   \~100 GB free disk at project start
-   Python 3.12.4
-   Ollama 0.32.15
-   Vision model: `qwen3-vl:4b-instruct`
-   Vehicle detector: `facebook/detr-resnet-50`
-   Python packages currently needed:
    -   `ollama`
    -   `Pillow`
    -   `torch`
    -   `torchvision`
    -   `transformers`
    -   `timm`

## Setup

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install ollama pillow torch torchvision transformers timm
```

Ollama model:

``` bash
ollama pull qwen3-vl:4b-instruct
```

## Current Project Structure

``` text
RaceSort/
├── .venv/
├── README.md
├── PROJECT_CONTEXT.md
├── DEVELOPMENT_LOG.md
├── test_qwen.py
├── test_detector.py
└── test-photos/
    ├── GGBM0021.JPG
    ├── GGBM0021-small.jpg
    ├── GGBM0021-detected.jpg
    └── GGBM0022.JPG
```

## Completed Proofs of Concept

### Phase 1 --- Local vision recognition

`test_qwen.py` opens `GGBM0021.JPG`, resizes it to a maximum 1500-pixel
dimension, sends it through Ollama to Qwen3-VL, and correctly receives
`706`.

``` text
Python → Pillow → Ollama → Qwen3-VL → 706
```

### Phase 2A --- Automatic motorcycle detection

`test_detector.py` uses DETR (`facebook/detr-resnet-50`) to detect
motorcycles in `GGBM0021.JPG`.

DETR detected 3 useful/visible motorcycles with confidence scores of
approximately:

-   0.99
-   1.00
-   0.98

Visual inspection confirmed all three bounding boxes were correct.

The photograph contains approximately five motorcycles in total, but two
are heavily obscured/background motorcycles. Their omission is currently
acceptable because they do not present useful visible race numbers.

This leads to an important evaluation principle:

> Detection success should be measured primarily by whether RaceSort
> finds every **sortable vehicle**---a vehicle sufficiently visible to
> identify and associate with a race number---not merely whether it
> detects every partially visible motorcycle or car.

## Next Milestone --- Phase 2B

Use the DETR bounding boxes to:

1.  Crop each detected motorcycle.
2.  Create analysis-sized crop images without modifying the original.
3.  Send each crop independently to Qwen3-VL.
4.  Ask for the exact visible race number or `UNKNOWN`.
5.  Print the results.

Target for `GGBM0021.JPG`:

``` text
Vehicle 1 → 49
Vehicle 2 → 706
Vehicle 3 → UNKNOWN
```

Do not add UI, databases, custom training, production export, or
dedicated OCR until this vehicle-detection → crop → number-recognition
pipeline is demonstrated and measured.

## Rules

-   Never modify/overwrite original photographs.
-   Use copies, proxies, or in-memory transformations.
-   Never force an AI guess when a number is unreadable.
-   Ambiguous results must eventually go to human review.
