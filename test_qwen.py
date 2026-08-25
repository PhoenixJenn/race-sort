from pathlib import Path
from PIL import Image
import ollama


IMAGE_PATH = Path("test-photos/GGBM0021.JPG")
RESIZED_PATH = Path("test-photos/GGBM0021-small.jpg")


# Open the original image
image = Image.open(IMAGE_PATH)

# Make a copy small enough for our local vision model.
image.thumbnail((1500, 1500))

# Save the smaller test image.
image.save(RESIZED_PATH, quality=90)

print(f"Created resized image: {RESIZED_PATH}")


prompt = """
Focus ONLY on the foreground black motorcycle with the large yellow
race-number plate.

What is the EXACT number printed on the plate?

Return only the number.
Do not explain your answer.
Do not guess.
"""


response = ollama.chat(
    model="qwen3-vl:4b-instruct",
    messages=[
        {
            "role": "user",
            "content": prompt,
            "images": [str(RESIZED_PATH)],
        }
    ],
)


print("Qwen response:")
print(response["message"]["content"])