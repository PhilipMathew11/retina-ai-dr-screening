import numpy as np
import pytest

from src.preprocessing import (
    crop_fundus_roi,
    extract_features,
    preprocess_fundus,
    prepare_model_input,
    validate_image,
)


def test_validate_converts_grayscale_to_rgb():
    image = np.full((64, 80), 100, dtype=np.uint8)
    result = validate_image(image)
    assert result.shape == (64, 80, 3)
    assert result.dtype == np.uint8


def test_validate_rejects_too_small_image():
    with pytest.raises(ValueError, match="at least 32"):
        validate_image(np.zeros((16, 16, 3), dtype=np.uint8))


def test_crop_fundus_removes_black_border():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[20:80, 25:95] = 180
    result = crop_fundus_roi(image)
    assert result.shape[0] < image.shape[0]
    assert result.shape[1] < image.shape[1]


def test_full_preprocessing_outputs_expected_shapes_and_features():
    image = np.zeros((160, 180, 3), dtype=np.uint8)
    image[15:145, 20:160] = (100, 130, 110)
    image[70:74, 30:150] = (30, 45, 35)
    processed = preprocess_fundus(image)
    assert processed.enhanced.shape == (224, 224, 3)
    assert processed.vessel_mask.shape == (224, 224)
    assert 0 <= processed.features["vessel_candidate_fraction"] <= 1
    assert prepare_model_input(image).shape == (224, 224, 3)


def test_feature_extraction_rejects_wrong_mask_shape():
    with pytest.raises(ValueError, match="dimensions"):
        extract_features(np.zeros((40, 40, 3), dtype=np.uint8), np.zeros((20, 20), dtype=np.uint8))
