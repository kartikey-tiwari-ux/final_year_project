from pathlib import Path

import torch
from PIL import Image

from src.preprocessing.image_preprocessing import (
    IMAGE_SIZE,
    load_image_safely,
    pretrained_transform,
)


def test_load_image_safely_returns_none_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    assert load_image_safely(missing) is None


def test_load_image_safely_returns_none_for_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not actually an image")
    assert load_image_safely(corrupt) is None


def test_load_image_safely_loads_valid_image(tmp_path):
    img_path = tmp_path / "valid.jpg"
    Image.new("RGB", (50, 50), color="red").save(img_path)
    img = load_image_safely(img_path)
    assert img is not None
    assert img.mode == "RGB"


def test_pretrained_transform_output_shape():
    img = Image.new("RGB", (100, 150), color="blue")
    tensor = pretrained_transform(img)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert isinstance(tensor, torch.Tensor)


def test_pretrained_transform_handles_grayscale_converted_image():
    img = Image.new("L", (80, 80), color=128).convert("RGB")
    tensor = pretrained_transform(img)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
