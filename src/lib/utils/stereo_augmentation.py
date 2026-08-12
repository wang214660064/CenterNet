"""不改变双目几何关系的轻量在线图像增强。"""

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np


def _clip_uint8(image):
  return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _shared_photometric(image, gamma, contrast, brightness, color_shift):
  """对一对图像使用完全相同的光照参数。"""
  value = image.astype(np.float32) / 255.0
  value = np.power(np.clip(value, 0, 1), gamma)
  value = (value - 0.5) * contrast + 0.5 + brightness
  value = value * color_shift.reshape(1, 1, 3)
  return _clip_uint8(value)


def augment_stereo_pair(left, right, rng=None, shared_prob=0.5,
                        mismatch_prob=0.15, strength=0.15):
  """
  增强左右目图像，但绝不改变尺寸和像素坐标。

  大部分光照变化对左右图同步施加；小概率只对一侧加入轻微
  曝光、噪声或模糊，模拟双目相机的曝光和成像质量差异。
  """
  if left is None or right is None:
    raise ValueError('左右目图像不能为空')
  if left.shape != right.shape:
    raise ValueError('左右目图像尺寸必须一致')
  if not 0 <= shared_prob <= 1 or not 0 <= mismatch_prob <= 1:
    raise ValueError('增强概率必须在0到1之间')
  if not 0 <= strength <= 0.3:
    raise ValueError('增强强度必须在0到0.3之间')

  rng = np.random if rng is None else rng
  left_aug, right_aug = left.copy(), right.copy()
  metadata = {'shared': False, 'mismatch': 'none', 'side': 'none'}

  if rng.random_sample() < shared_prob:
    gamma = rng.uniform(1.0 - strength, 1.0 + strength)
    contrast = rng.uniform(1.0 - strength, 1.0 + strength)
    brightness = rng.uniform(-0.35 * strength, 0.35 * strength)
    temperature = rng.uniform(-0.35 * strength, 0.35 * strength)
    # OpenCV通道顺序为BGR，蓝红反向微调可模拟轻微色温变化。
    color_shift = np.asarray(
        [1.0 - temperature, 1.0, 1.0 + temperature], dtype=np.float32)
    left_aug = _shared_photometric(
        left_aug, gamma, contrast, brightness, color_shift)
    right_aug = _shared_photometric(
        right_aug, gamma, contrast, brightness, color_shift)
    metadata['shared'] = True

  if rng.random_sample() < mismatch_prob:
    side = 'left' if rng.random_sample() < 0.5 else 'right'
    target = left_aug if side == 'left' else right_aug
    mode = ('exposure', 'noise', 'blur')[rng.randint(0, 3)]
    if mode == 'exposure':
      factor = rng.uniform(1.0 - 0.6 * strength, 1.0 + 0.6 * strength)
      target = np.clip(target.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    elif mode == 'noise':
      sigma = rng.uniform(1.0, max(1.1, 30.0 * strength))
      noise = rng.normal(0, sigma, target.shape).astype(np.float32)
      target = np.clip(target.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
      target = cv2.GaussianBlur(target, (3, 3), rng.uniform(0.4, 0.9))
    if side == 'left':
      left_aug = target
    else:
      right_aug = target
    metadata.update({'mismatch': mode, 'side': side})

  return left_aug, right_aug, metadata
