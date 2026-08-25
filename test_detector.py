from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection


IMAGE_PATH = Path("test-photos/GGBM0021.JPG")
OUTPUT_PATH = Path("test-photos/GGBM0021-detected.jpg")

MODEL_NAME = "facebook/detr-resnet-50"

# Load the pretrained DETR image processor and model.
processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
model = DetrForObjectDetection.from_pretrained(MODEL_NAME)

# Put the model in evaluation mode.
model.eval()

# Open the original photograph.
image = Image.open(IMAGE_PATH).convert("RGB")

# Convert the image into the tensor format DETR expects.
inputs = processor(images=image, return_tensors="pt")

# Run object detection.
with torch.no_grad():
    outputs = model(**inputs)

# Convert model output back into normal image coordinates.
target_sizes = torch.tensor([image.size[::-1]])

results = processor.post_process_object_detection(
    outputs,
    target_sizes=target_sizes,
    threshold=0.7,
)[0]

# Draw detections onto a COPY of the image.
debug_image = image.copy()
draw = ImageDraw.Draw(debug_image)

motorcycle_count = 0

for score, label, box in zip(
    results["scores"],
    results["labels"],
    results["boxes"],
):
    class_name = model.config.id2label[label.item()]
    confidence = score.item()

    if class_name == "motorcycle":
        motorcycle_count += 1

        x1, y1, x2, y2 = box.tolist()

        draw.rectangle(
            (x1, y1, x2, y2),
            width=6,
        )

        draw.text(
            (x1, max(0, y1 - 20)),
            f"motorcycle {confidence:.2f}",
        )

        print(
            f"Motorcycle {motorcycle_count}: "
            f"confidence={confidence:.2f}, "
            f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )

debug_image.save(OUTPUT_PATH, quality=95)

print()
print(f"Motorcycles detected: {motorcycle_count}")
print(f"Saved debug image: {OUTPUT_PATH}")