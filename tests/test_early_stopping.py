import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / 'src' / 'main.py'
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location('centernet_main', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_early_stopping_resets_counter_when_metric_improves():
  best, count, improved = MODULE.update_early_stopping(3.0, 4.0, 0.01, 2)
  assert (best, count, improved) == (3.0, 0, True)


def test_early_stopping_counts_only_when_metric_does_not_improve():
  best, count, improved = MODULE.update_early_stopping(3.995, 4.0, 0.01, 2)
  assert (best, count, improved) == (4.0, 3, False)
