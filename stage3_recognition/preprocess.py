"""Prepare a crop for the recognizers.

Two steps only, because both models resize and normalise images themselves:
1. Flatten transparency onto a white background and convert to 3-channel RGB.
2. Add a white margin (``pad_in`` inches, converted with the crop's dpi), because
   stage 1-2 cuts boxes tightly and an atom label touching the edge is easy to lose.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from .config import PreprocessConfig
from .handoff import CropTask


def to_rgb_on_background(image: Image.Image, background: int = 255) -> Image.Image:
    """Return an RGB image, putting any transparent pixels on a plain background."""
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (background, background, background, 255))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return image.convert("RGB")


def pad_pixels(pad_in: float, dpi: int) -> int:
    """Margin size in pixels for a crop rendered at ``dpi``."""
    return max(0, round(pad_in * dpi))


def prepare_image(image: Image.Image, dpi: int, cfg: PreprocessConfig) -> Image.Image:
    rgb = to_rgb_on_background(image, cfg.background)
    pad = pad_pixels(cfg.pad_in, dpi)
    if pad == 0:
        return rgb
    fill = (cfg.background,) * 3
    return ImageOps.expand(rgb, border=pad, fill=fill)


def prepare_crop(task: CropTask, cfg: PreprocessConfig, out_dir: Path) -> Path:
    """Prepare one crop and save it; returns the path the recognizers will read."""
    with Image.open(task.image_path) as image:
        prepared = prepare_image(image, task.dpi, cfg)
    target = Path(out_dir) / f"paper_{task.paper_id}" / f"{task.crop_id}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    prepared.save(target)
    return target
