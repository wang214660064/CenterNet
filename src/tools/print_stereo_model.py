#!/usr/bin/env python3
"""打印stereo_ddd模型结构、输出头和参数规模。"""

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'lib'))

from models.model import create_model  # noqa: E402
from opts import opts  # noqa: E402


def main():
  option = opts().init([
      'stereo_ddd', '--arch', 'stereo_dla_34', '--gpus', '-1'])
  model = create_model(option.arch, option.heads, option.head_conv)
  model.eval()
  total = sum(parameter.numel() for parameter in model.parameters())
  trainable = sum(parameter.numel() for parameter in model.parameters()
                  if parameter.requires_grad)
  print(model)
  print('\n输出头: {}'.format(option.heads))
  print('总参数量: {:,}'.format(total))
  print('可训练参数量: {:,}'.format(trainable))
  with torch.no_grad():
    output = model(
        torch.zeros(1, 3, 384, 1280),
        torch.full((1, 1, 96, 320), 20.0),
        torch.ones(1, 1, 96, 320))[-1]
  print('输出形状:')
  for name, tensor in output.items():
    print('  {}: {}'.format(name, tuple(tensor.shape)))


if __name__ == '__main__':
  main()
