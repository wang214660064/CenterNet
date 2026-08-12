from pathlib import Path
import sys

import numpy as np
import pytest


LIB = Path(__file__).parents[1] / 'src' / 'lib'
sys.path.insert(0, str(LIB))

from utils.backbone_augmentation import augment_backbone_image


def sample_image():
  x = np.linspace(20, 220, 32, dtype=np.uint8)
  return np.tile(x[None, :, None], (24, 1, 3))


def test_disabled_augmentation_keeps_structure_and_pixels():
  image = sample_image()
  result, applied = augment_backbone_image(
      image, rng=np.random.RandomState(1), probability=0)
  assert not applied
  assert result.shape == image.shape
  assert result.dtype == image.dtype
  assert np.array_equal(result, image)
  assert result is not image


def test_enabled_augmentation_changes_only_appearance():
  image = sample_image()
  result, applied = augment_backbone_image(
      image, rng=np.random.RandomState(2), probability=1, strength=0.15)
  assert applied
  assert result.shape == image.shape
  assert result.dtype == image.dtype
  assert not np.array_equal(result, image)


def test_augmentation_is_reproducible_with_fixed_seed():
  image = sample_image()
  first, _ = augment_backbone_image(
      image, rng=np.random.RandomState(317), probability=1)
  second, _ = augment_backbone_image(
      image, rng=np.random.RandomState(317), probability=1)
  assert np.array_equal(first, second)


def test_rejects_invalid_strength():
  with pytest.raises(ValueError, match='强度'):
    augment_backbone_image(sample_image(), probability=1, strength=0.5)
