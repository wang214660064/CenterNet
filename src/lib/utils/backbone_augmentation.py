"""只作用于Backbone图像分支的轻量外观增强。"""

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np


def augment_backbone_image(image, rng=None, probability=0.5, strength=0.15):
  """
  改变亮度、对比度、Gamma、饱和度和轻微色温。

  函数不改变图像尺寸、像素坐标和数据类型，因此不会改变
  2D/3D标注和标定。SGBM必须继续使用原始左右图计算。
  """
  if image is None:
    raise ValueError('Backbone输入图像不能为空')
  if image.ndim != 3 or image.shape[2] != 3:
    raise ValueError('Backbone输入必须是H×W×3彩色图像')
  if not 0 <= probability <= 1:
    raise ValueError('增强概率必须在0到1之间')
  if not 0 <= strength <= 0.3:
    raise ValueError('增强强度必须在0到0.3之间')

  rng = np.random if rng is None else rng
  if rng.random_sample() >= probability or strength == 0:
    return image.copy(), False

  value = image.astype(np.float32) / 255.0
  gamma = rng.uniform(1.0 - strength, 1.0 + strength)
  contrast = rng.uniform(1.0 - strength, 1.0 + strength)
  brightness = rng.uniform(-0.3 * strength, 0.3 * strength)
  value = np.power(np.clip(value, 0, 1), gamma)
  value = (value - 0.5) * contrast + 0.5 + brightness

  # HSV中只做轻微饱和度调整，不改变边缘和目标几何。
  value = np.clip(value * 255.0, 0, 255).astype(np.uint8)
  hsv = cv2.cvtColor(value, cv2.COLOR_BGR2HSV).astype(np.float32)
  hsv[:, :, 1] *= rng.uniform(1.0 - strength, 1.0 + strength)
  hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
  value = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

  # OpenCV使用BGR通道，蓝红反向微调模拟轻微色温变化。
  temperature = rng.uniform(-0.25 * strength, 0.25 * strength)
  channel_scale = np.asarray(
      [1.0 - temperature, 1.0, 1.0 + temperature], dtype=np.float32)
  value = np.clip(
      value.astype(np.float32) * channel_scale.reshape(1, 1, 3),
      0, 255).astype(np.uint8)
  return value, True
