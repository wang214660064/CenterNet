from pathlib import Path
import sys

import numpy as np
import pytest


LIB = Path(__file__).parents[1] / 'src' / 'lib'
sys.path.insert(0, str(LIB))

from utils.stereo_augmentation import augment_stereo_pair
from datasets.sample.stereo_ddd import StereoDddDataset


def sample_pair():
  x = np.linspace(20, 220, 32, dtype=np.uint8)
  left = np.tile(x[None, :, None], (24, 1, 3))
  right = np.roll(left, 2, axis=1)
  return left, right


def test_disabled_augmentation_keeps_pixels_and_geometry():
  left, right = sample_pair()
  left_aug, right_aug, metadata = augment_stereo_pair(
      left, right, rng=np.random.RandomState(1),
      shared_prob=0, mismatch_prob=0)

  assert np.array_equal(left_aug, left)
  assert np.array_equal(right_aug, right)
  assert left_aug.shape == left.shape
  assert right_aug.shape == right.shape
  assert metadata == {'shared': False, 'mismatch': 'none', 'side': 'none'}


def test_shared_augmentation_uses_same_photometric_mapping():
  left, _ = sample_pair()
  left_aug, right_aug, metadata = augment_stereo_pair(
      left, left.copy(), rng=np.random.RandomState(2),
      shared_prob=1, mismatch_prob=0)

  assert metadata['shared']
  assert np.array_equal(left_aug, right_aug)
  assert not np.array_equal(left_aug, left)


def test_camera_mismatch_changes_only_one_side():
  left, right = sample_pair()
  left_aug, right_aug, metadata = augment_stereo_pair(
      left, right, rng=np.random.RandomState(3),
      shared_prob=0, mismatch_prob=1)

  changed = [not np.array_equal(left_aug, left),
             not np.array_equal(right_aug, right)]
  assert sum(changed) == 1
  assert metadata['mismatch'] in {'exposure', 'noise', 'blur'}
  assert metadata['side'] in {'left', 'right'}


def test_rejects_different_stereo_shapes():
  left, right = sample_pair()
  with pytest.raises(ValueError, match='尺寸必须一致'):
    augment_stereo_pair(left, right[:, :-1])


def test_dataset_reuses_augmented_left_image_across_kitti_paths():
  left, _ = sample_pair()
  dataset = StereoDddDataset.__new__(StereoDddDataset)
  dataset._stereo_left_path = '/data/kitti/training/image_2/000008.png'
  dataset._stereo_left_image = left

  result = dataset._read_image('/data/kitti/images/trainval/000008.png')
  assert result is left
