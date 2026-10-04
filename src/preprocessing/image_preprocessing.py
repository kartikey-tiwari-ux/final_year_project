"""Image preprocessing for the image pipeline (see PROJECT_ARCHITECTURE.md)."""
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Used for the pretrained (ResNet18) branch, which expects ImageNet-style normalization.
pretrained_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Used for the from-scratch baseline CNN (no pretrained-specific normalization needed).
baseline_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

# Applied only to the training split (see REPRODUCIBILITY.md: augmentation must not
# touch validation/test).
train_augmentation = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ]
)


def load_image_safely(path: Path) -> Optional[Image.Image]:
    """Load and RGB-convert an image, returning None (not raising) for missing/corrupt
    files so the caller can filter and log how many samples were dropped."""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        img.load()
        return img
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None
