"""Validated image-quality measurements and routing filters."""

from PIL import Image, ImageFilter, ImageStat


def measure_sharpness(image_path):
    """Return the validated 800px edge-variance sharpness score."""

    with Image.open(image_path) as source:
        image = source.convert("L")
    image.thumbnail((800, 800))
    edges = image.filter(ImageFilter.FIND_EDGES)
    return ImageStat.Stat(edges).var[0]


def should_filter_non_primary(
    vehicles_in_photo,
    relative_area,
    relative_sharpness,
    max_relative_area,
    max_relative_sharpness,
):
    """Apply the validated conservative non-primary rule."""

    return (
        vehicles_in_photo > 1
        and relative_area < max_relative_area
        and relative_sharpness < max_relative_sharpness
    )


def should_filter_too_blurry(sharpness, max_sharpness):
    """Apply the validated conservative absolute-blur rule."""

    return sharpness < max_sharpness
