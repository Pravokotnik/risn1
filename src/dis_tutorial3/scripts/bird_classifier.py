#!/usr/bin/env python3

# bird_classifier.py

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import numpy as np
import sys

# Get the package path
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ─── 1. Adjust these two paths to point to your checkpoint and mapping files ─────────
CHECKPOINT_PATH = os.path.join("bird_checkpoints_updated", "best_model_updated.pth")
MAPPING_PATH    = os.path.join("bird_checkpoints_updated", "idx_to_class_updated.pth")

# ─── Constants ───────────────────────────────────────────────────────────────────────
INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define model class (matching the architecture used during training)
class BirdClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BirdClassifier, self).__init__()
        # Using a pretrained ResNet backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


# ─── 2. Model Loading ─────────────────────────────────────────────────────────────────
def load_model_and_mapping(checkpoint_path: str, mapping_path: str, device: torch.device):
    """
    Loads a ResNet-50 model fine-tuned on bird species, plus the idx->class mapping.
    Returns (model, idx_to_class_dict).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at: {mapping_path}")

    idx_to_class = torch.load(mapping_path, map_location=device)  # e.g. {0: "Laysan_Albatross", …}
    num_classes = len(idx_to_class)

    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, idx_to_class


# Load once at import time
_MODEL, _IDX_TO_CLASS = load_model_and_mapping(CHECKPOINT_PATH, MAPPING_PATH, DEVICE)


# ─── 3. Cropping Helpers ──────────────────────────────────────────────────────────────
def crop_bird_region(img_bgr: np.ndarray) -> Image.Image | None:
    """
    1) Convert to HSV, threshold for blue (assumes ring is blue) to isolate the ring.
    2) Find largest contour, fit min enclosing circle (cx, cy, r).
    3) Crop a box just above the ring: left/right = cx ± 0.8r; top = cy - 1.5r; bottom = cy - 0.2r.
    4) If any step fails, return None, so caller can fall back to other crops.
    """
    height, width = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Blue-range in HSV (tweak if needed)
    lower_blue = np.array([100, 120,  60])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None

    (cx, cy), r = cv2.minEnclosingCircle(largest)
    cx, cy, r = float(cx), float(cy), float(r)
    if r < 10 or r > min(width, height) / 2:
        return None

    left   = int(max(0, cx - 0.8 * r))
    right  = int(min(width, cx + 0.8 * r))
    top    = int(max(0, cy - 1.5 * r))
    bottom = int(min(height, cy - 0.2 * r))

    if left >= right or top >= bottom:
        return None

    rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return rgb_pil.crop((left, top, right, bottom))


def top_center_crop(img_bgr: np.ndarray) -> Image.Image:
    """
    224×224 crop centered horizontally, ¼ down from top.
    Always returns a PIL.Image; pad/calc to ensure 224×224.
    """
    height, width = img_bgr.shape[:2]
    crop_size = INPUT_SIZE
    cx = width // 2
    cy = height // 4

    left   = max(0, cx - crop_size // 2)
    right  = min(width, cx + crop_size // 2)
    top    = max(0, cy - crop_size // 2)
    bottom = min(height, top + crop_size)

    # Adjust if image is too small
    if right - left < crop_size:
        left = max(0, right - crop_size)
    if bottom - top < crop_size:
        top = max(0, bottom - crop_size)

    rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return rgb_pil.crop((left, top, right, bottom))


def center_crop_full(img_bgr: np.ndarray) -> Image.Image:
    """
    224×224 center crop of the entire frame.
    """
    height, width = img_bgr.shape[:2]
    crop_size = INPUT_SIZE

    left   = max(0, width // 2 - crop_size // 2)
    top    = max(0, height // 2 - crop_size // 2)
    right  = min(width, left + crop_size)
    bottom = min(height, top + crop_size)

    # Adjust if needed
    if right - left < crop_size:
        left = max(0, right - crop_size)
    if bottom - top < crop_size:
        top = max(0, bottom - crop_size)

    rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return rgb_pil.crop((left, top, right, bottom))


# ─── 4. Preprocessing Pipeline ────────────────────────────────────────────────────────
_preproc = transforms.Compose([
    transforms.Resize(int(INPUT_SIZE * 1.14)),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_topk_on_crop(model: torch.nn.Module,
                         idx_to_class: dict[int, str],
                         pil_crop: Image.Image,
                         topk: int,
                         device: torch.device) -> list[tuple[str, float]]:
    """
    Given a PIL.Image crop, run it through the model and return a list of
    (species_name, probability) for the top‐k predictions.
    """
    tensor = _preproc(pil_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)

    top_probs, top_idxs = probs.topk(topk, dim=1)
    top_probs = top_probs.cpu().squeeze(0)
    top_idxs  = top_idxs.cpu().squeeze(0)

    results = []
    for idx, p in zip(top_idxs, top_probs):
        species = idx_to_class[int(idx)]
        results.append((species, float(p.item())))
    return results


def predict_topk(img_bgr: np.ndarray, topk: int = 3) -> list[tuple[str, float]]:
    """
    Given a single OpenCV image (BGR NumPy array), run all crops (HSV‐based ring crop,
    top‐center crop, center crop), pick the crop whose top‐1 confidence is highest,
    then return the top‐k predictions (species + probability) from that best crop.
    """
    # 1. Generate candidate crops
    crops: list[Image.Image] = []
    hsv_crop = crop_bird_region(img_bgr)
    if hsv_crop is not None:
        crops.append(hsv_crop)

    # Fallback crops
    crops.append(top_center_crop(img_bgr))
    crops.append(center_crop_full(img_bgr))

    # 2. Choose the crop with highest top‐1 confidence
    best_conf = -1.0
    best_crop: Image.Image | None = None
    for crop in crops:
        top1, prob = predict_topk_on_crop(_MODEL, _IDX_TO_CLASS, crop, 1, DEVICE)[0]
        if prob > best_conf:
            best_conf = prob
            best_crop = crop

    # 3. Now get top‐k predictions on best crop
    if best_crop is None:
        raise RuntimeError("Failed to generate any valid crop for bird classification.")
    return predict_topk_on_crop(_MODEL, _IDX_TO_CLASS, best_crop, topk, DEVICE)


def predict_bird_name(img_bgr: np.ndarray) -> str:
    """
    Shortcut: return only the top‐1 species string for the given OpenCV image.
    """
    top1_species, _ = predict_topk(img_bgr, topk=1)[0]
    return top1_species


if __name__ == "__main__":
    # Example usage when run standalone:
    import glob

    image_paths = glob.glob(os.path.join("images", "*"))
    image_paths.sort()

    if not image_paths:
        print("No images found under 'images/'. Exiting.")
        exit(0)

    for img_path in image_paths:
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        top3 = predict_topk(bgr, topk=3)
        print(f"\nImage: {img_path}")
        for rank, (species, prob) in enumerate(top3, start=1):
            print(f"  {rank}. {species:<30} {prob*100:5.2f}%")
