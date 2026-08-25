from pathlib import Path

import ollama
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


IMAGE_PATH = Path("test-photos/GGBM0021.JPG")
CROPS_DIR = Path("test-photos/crops")

DETECTOR_MODEL = "facebook/detr-resnet-50"
VISION_MODEL = "qwen3-vl:4b-instruct"

DETECTION_THRESHOLD = 0.70
MAX_CROP_SIZE = 1500


CROPS_DIR.mkdir(exist_ok=True)

print("Loading DETR...")

processor = DetrImageProcessor.from_pretrained(DETECTOR_MODEL)
detector = DetrForObjectDetection.from_pretrained(DETECTOR_MODEL)
detector.eval()

print("Opening image...")

image = Image.open(IMAGE_PATH).convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = detector(**inputs)

target_sizes = torch.tensor([image.size[::-1]])

results = processor.post_process_object_detection(
    outputs,
    target_sizes=target_sizes,
    threshold=DETECTION_THRESHOLD,
)[0]


motorcycles = []

for score, label, box in zip(
    results["scores"],
    results["labels"],
    results["boxes"],
):
    class_name = detector.config.id2label[label.item()]

    if class_name == "motorcycle":
        motorcycles.append(
            {
                "score": score.item(),
                "box": box.tolist(),
            }
        )


print()
print(f"Motorcycles detected: {len(motorcycles)}")
print()


for index, motorcycle in enumerate(motorcycles, start=1):

    x1, y1, x2, y2 = motorcycle["box"]

    # Pillow needs integer pixel coordinates.
    crop_box = (
        int(x1),
        int(y1),
        int(x2),
        int(y2),
    )

    crop = image.crop(crop_box)

    # Keep the image small enough for Qwen's vision context.
    crop.thumbnail(
        (MAX_CROP_SIZE, MAX_CROP_SIZE)
    )

    crop_path = CROPS_DIR / f"motorcycle-{index}.jpg"

    crop.save(
        crop_path,
        quality=95,
    )

    print(
        f"Motorcycle {index}: "
        f"DETR confidence={motorcycle['score']:.2f}"
    )

    print(
        f"Saved crop: {crop_path}"
    )

    prompt = """
This image contains one race motorcycle.

Look carefully for the motorcycle's race number.

Return ONLY the race number if you can clearly read it.

The race number may contain one or more digits.

Do not guess.
Do not explain your reasoning.

If no race number can be read confidently, return exactly:

UNKNOWN
"""

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(crop_path)],
            }
        ],
    )

    race_number = response["message"]["content"].strip()

    print(
        f"Qwen result: {race_number}"
    )

    print("-" * 40)