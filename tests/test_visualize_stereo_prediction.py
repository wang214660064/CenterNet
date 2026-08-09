import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / 'src' / 'tools' / 'visualize_stereo_prediction.py'
SPEC = importlib.util.spec_from_file_location('visualize_stereo_prediction', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_find_image_index():
  assert MODULE.find_image_index([0, 8, 15], '000008') == 1
  with pytest.raises(ValueError):
    MODULE.find_image_index([0, 8, 15], '000009')
