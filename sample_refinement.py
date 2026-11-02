
from unstructured.partition.image import partition_image
from PIL import Image, ImageDraw, ImageFont
import os

# Input image
image_path = (
    r"D:\AI Engineering\HeyArro\screen-segments"
    r"\screenshot-20261102-141716-87edfa\screenshot.png"
)

# Output image
output_path = os.path.join(
    os.path.dirname(image_path),
    "screenshot_bounding_boxes.png"
)

# Extract elements from the image
elements = partition_image(filename=image_path)

# Open the original image
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# Font for labels
try:
    font = ImageFont.truetype("arial.ttf", 16)
except OSError:
    font = ImageFont.load_default()

# Track successfully drawn bounding boxes
boxes_drawn = 0

for index, element in enumerate(elements, start=1):
    metadata = element.metadata

    # Unstructured may store coordinates in metadata.coordinates
    coordinates = getattr(metadata, "coordinates", None)

    if not coordinates:
        print(
            f"Element {index}: {element.category} "
            "— no coordinates available"
        )
        continue

    points = coordinates.points
    if not points:
        continue

    # Coordinates are typically polygon points.
    # Calculate the enclosing rectangle.
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    x1 = int(min(x_coords))
    y1 = int(min(y_coords))
    x2 = int(max(x_coords))
    y2 = int(max(y_coords))

    # Draw bounding box
    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=3
    )

    # Label with element category
    label = f"{index}: {element.category}"
    draw.text(
        (x1, max(0, y1 - 20)),
        label,
        fill="red",
        font=font,
        stroke_width=1,
        stroke_fill="white"
    )

    boxes_drawn += 1

# Save annotated image
image.save(output_path)

print(f"Total elements extracted: {len(elements)}")
print(f"Bounding boxes drawn: {boxes_drawn}")
print(f"Output saved to: {output_path}")

# Print extracted elements for debugging
for index, element in enumerate(elements, start=1):
    print(
        index,
        element.category,
        repr(str(element)[:150]),
        element.metadata.coordinates
        if element.metadata
        else None
    )
