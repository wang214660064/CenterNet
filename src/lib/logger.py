from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import csv
import json
import torch
import shutil
USE_TENSORBOARD = True
try:
  import tensorboardX
  print('Using tensorboardX')
except:
  USE_TENSORBOARD = False

class Logger(object):
  def __init__(self, opt):
    """Create a summary writer logging to log_dir."""
    if not os.path.exists(opt.save_dir):
      os.makedirs(opt.save_dir)
    if not os.path.exists(opt.debug_dir):
      os.makedirs(opt.debug_dir)
   
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    args = dict((name, getattr(opt, name)) for name in dir(opt)
                if not name.startswith('_'))
    file_name = os.path.join(opt.save_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> torch version: {}\n'.format(torch.__version__))
      opt_file.write('==> cudnn version: {}\n'.format(
        torch.backends.cudnn.version()))
      opt_file.write('==> Cmd:\n')
      opt_file.write(str(sys.argv))
      opt_file.write('\n==> Opt:\n')
      for k, v in sorted(args.items()):
        opt_file.write('  %s: %s\n' % (str(k), str(v)))
          
    log_dir = opt.save_dir + '/logs_{}'.format(time_str)
    if USE_TENSORBOARD:
      self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
    else:
      if not os.path.exists(os.path.dirname(log_dir)):
        os.mkdir(os.path.dirname(log_dir))
      if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    self.log = open(log_dir + '/log.txt', 'w')
    shutil.copy2(file_name, log_dir)
    self.start_line = True
    self.results_path = os.path.join(opt.save_dir, 'results.csv')
    self.plot_path = os.path.join(opt.save_dir, 'results.png')
    self.summary_path = os.path.join(opt.save_dir, 'training_summary.json')
    self.history = self._load_history()

  def write(self, txt):
    if self.start_line:
      time_str = time.strftime('%Y-%m-%d-%H-%M')
      self.log.write('{}: {}'.format(time_str, txt))
    else:
      self.log.write(txt)  
    self.start_line = False
    if '\n' in txt:
      self.start_line = True
      self.log.flush()
  
  def close(self):
    self.log.close()
    if USE_TENSORBOARD:
      self.writer.close()
  
  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    if USE_TENSORBOARD:
      self.writer.add_scalar(tag, value, step)

  def _load_history(self):
    if not os.path.exists(self.results_path):
      return []
    with open(self.results_path, 'r', newline='') as results_file:
      return list(csv.DictReader(results_file))

  @staticmethod
  def _number(value):
    if isinstance(value, torch.Tensor):
      value = value.detach().cpu().item()
    return float(value)

  def log_epoch(self, epoch, lr, train_metrics, val_metrics=None, best=None):
    row = {'epoch': int(epoch), 'lr': self._number(lr)}
    for name, value in train_metrics.items():
      row['train/{}'.format(name)] = self._number(value)
    if val_metrics:
      for name, value in val_metrics.items():
        row['val/{}'.format(name)] = self._number(value)

    self.history = [item for item in self.history
                    if int(float(item['epoch'])) != int(epoch)]
    self.history.append(row)
    self.history.sort(key=lambda item: int(float(item['epoch'])))
    fieldnames = ['epoch', 'lr']
    for item in self.history:
      for name in item:
        if name not in fieldnames:
          fieldnames.append(name)
    with open(self.results_path, 'w', newline='') as results_file:
      writer = csv.DictWriter(results_file, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(self.history)

    summary = {
        'latest_epoch': int(epoch),
        'latest_lr': self._number(lr),
        'latest_train_metrics': {
            name: self._number(value) for name, value in train_metrics.items()},
        'latest_val_metrics': {
            name: self._number(value) for name, value in (val_metrics or {}).items()},
        'best_metric': None if best is None else self._number(best),
        'artifacts': {
            'metrics': self.results_path,
            'curves': self.plot_path,
        },
    }
    with open(self.summary_path, 'w', encoding='utf-8') as summary_file:
      json.dump(summary, summary_file, indent=2, ensure_ascii=False)
    self.plot_results()

  def plot_results(self):
    try:
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
    except ImportError:
      return

    groups = [
        ('Total loss', ['train/loss', 'val/loss']),
        ('Depth', ['train/dep_loss', 'train/depth_offset_loss',
                'train/depth_gate_loss', 'train/depth_fusion_loss',
                'val/dep_loss', 'val/depth_offset_loss',
                'val/depth_gate_loss', 'val/depth_fusion_loss']),
        ('2D detection', ['train/hm_loss', 'train/wh_loss', 'train/off_loss',
                   'val/hm_loss', 'val/wh_loss', 'val/off_loss']),
        ('3D attributes', ['train/dim_loss', 'train/rot_loss',
                   'train/proj_center_loss', 'train/proj_center_xy_loss',
                   'val/dim_loss', 'val/rot_loss', 'val/proj_center_loss',
                   'val/proj_center_xy_loss']),
    ]
    epochs = [int(float(item['epoch'])) for item in self.history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (title, names) in zip(axes.flat, groups):
      has_line = False
      for name in names:
        values = []
        valid_epochs = []
        for epoch, item in zip(epochs, self.history):
          value = item.get(name, '')
          if value not in ('', None):
            valid_epochs.append(epoch)
            values.append(float(value))
        if values:
          axis.plot(valid_epochs, values, marker='o', markersize=3,
                    linewidth=1.5, label=name)
          has_line = True
      axis.set_title(title)
      axis.set_xlabel('Epoch')
      axis.grid(True, alpha=0.25)
      if has_line:
        axis.legend(fontsize=8)
    fig.suptitle('Stereo CenterNet Training Metrics', fontsize=15)
    fig.savefig(self.plot_path, dpi=160)
    plt.close(fig)
