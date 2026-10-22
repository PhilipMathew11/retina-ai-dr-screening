"""Fundus image validation and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ProcessedFundus:
    original: np.ndarray
    cropped: np.ndarray
    enhanced: np.ndarray
    vessel_mask: np.ndarray
    features: dict[str, float]


def validate_image(image: np.ndarray) -> np.ndarray:
    """Validate an in-memory image and return it as an RGB uint8 array."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("The uploaded file is not a readable image.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError("The image must have one, three, or four channels.")
    if min(image.shape[:2]) < 32:
        raise ValueError("Image dimensions must be at least 32 by 32 pixels.")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def crop_fundus_roi(image_rgb: np.ndarray) -> np.ndarray:
    """Crop the non-black fundus region, falling back safely if no ROI is found."""
    image_rgb = validate_image(image_rgb)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    roi = gray > 10
    ys, xs = np.where(roi)
    if xs.size == 0:
        return image_rgb.copy()
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    pad_x, pad_y = max(1, int(width * 0.01)), max(1, int(height * 0.01))
    x0, x1 = max(0, x0 - pad_x), min(image_rgb.shape[1], x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(image_rgb.shape[0], y1 + pad_y)
    crop = image_rgb[y0:y1, x0:x1]
    return crop if min(crop.shape[:2]) >= 32 else image_rgb.copy()


def enhance_contrast(image_rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE to LAB lightness while preserving chromatic channels."""
    image_rgb = validate_image(image_rgb)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(lightness), a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def segment_vessels(image_rgb: np.ndarray) -> np.ndarray:
    """Create a classical green-channel vessel-candidate mask for demonstration."""
    image_rgb = validate_image(image_rgb)
    green = image_rgb[:, :, 1]
    green = cv2.GaussianBlur(green, (3, 3), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)
    background = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    vessel_candidates = cv2.subtract(background, enhanced)
    _, mask = cv2.threshold(
        vessel_candidates, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_kernel)


def extract_features(image_rgb: np.ndarray, vessel_mask: np.ndarray) -> dict[str, float]:
    """Calculate simple descriptive image features for educational inspection."""
    image_rgb = validate_image(image_rgb)
    if vessel_mask.shape != image_rgb.shape[:2]:
        raise ValueError("Vessel mask dimensions must match the image.")
    normalized = image_rgb.astype(np.float32) / 255.0
    return {
        "mean_red": float(normalized[:, :, 0].mean()),
        "mean_green": float(normalized[:, :, 1].mean()),
        "mean_blue": float(normalized[:, :, 2].mean()),
        "intensity_std": float(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).std() / 255),
        "vessel_candidate_fraction": float(np.count_nonzero(vessel_mask) / vessel_mask.size),
    }


def preprocess_fundus(image_rgb: np.ndarray) -> ProcessedFundus:
    """Run display-oriented DIP steps and return intermediate outputs."""
    original = validate_image(image_rgb)
    cropped = crop_fundus_roi(original)
    denoised = cv2.GaussianBlur(cropped, (3, 3), 0)
    enhanced = enhance_contrast(denoised)
    resized = cv2.resize(enhanced, (224, 224), interpolation=cv2.INTER_AREA)
    vessel_mask = segment_vessels(resized)
    features = extract_features(resized, vessel_mask)
    return ProcessedFundus(
        original=original,
        cropped=cropped,
        enhanced=resized,
        vessel_mask=vessel_mask,
        features=features,
    )


def prepare_model_input(image_rgb: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Apply the same ROI, mild denoising, and CLAHE steps used by training."""
    image = crop_fundus_roi(validate_image(image_rgb))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = enhance_contrast(image)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)
