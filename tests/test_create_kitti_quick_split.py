import copy
import importlib.util
from pathlib import Path
import random


SCRIPT = Path(__file__).parents[1] / 'src' / 'tools' / 'create_kitti_project_split.py'
SPEC = importlib.util.spec_from_file_location('create_kitti_project_split', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_subset_is_deterministic_and_filters_annotations():
  dataset = {
      'images': [{'id': value} for value in range(10)],
      'annotations': [{'id': value + 1, 'image_id': value} for value in range(10)],
      'categories': [{'id': 1, 'name': 'Car'}],
  }
  first = MODULE.select_subset(copy.deepcopy(dataset), 4, random.Random(317))
  second = MODULE.select_subset(copy.deepcopy(dataset), 4, random.Random(317))
  assert first == second
  ids = {image['id'] for image in first['images']}
  assert len(ids) == 4
  assert {ann['image_id'] for ann in first['annotations']} == ids
