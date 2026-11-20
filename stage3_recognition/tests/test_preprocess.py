from PIL import Image

from stage3_recognition.config import PreprocessConfig
from stage3_recognition.preprocess import pad_pixels, prepare_image


def test_margin_is_converted_from_inches_with_the_dpi():
    assert pad_pixels(0.05, 300) == 15
    assert pad_pixels(0.05, 150) == 8
    assert pad_pixels(0.0, 300) == 0


def test_transparent_crop_becomes_rgb_on_white_with_margin():
    crop = Image.new("RGBA", (40, 20), (0, 0, 0, 0))          # fully transparent
    out = prepare_image(crop, dpi=300, cfg=PreprocessConfig(pad_in=0.05))
    assert out.mode == "RGB"
    assert out.size == (40 + 30, 20 + 30)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((35, 25)) == (255, 255, 255)          # transparency flattened to white


def test_margin_can_be_switched_off():
    crop = Image.new("L", (10, 10), 0)
    out = prepare_image(crop, dpi=300, cfg=PreprocessConfig(pad_in=0.0))
    assert out.size == (10, 10) and out.mode == "RGB"
