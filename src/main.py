from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def update_early_stopping(current_metric, best_metric, min_delta,
                          validations_without_improvement):
  """更新早停状态；只有验证指标未改善时才增加计数。"""
  improved = current_metric < best_metric - min_delta
  if improved:
    return current_metric, 0, True
  return best_metric, validations_without_improvement + 1, False


def configure_stereo_only_training(model):
  """冻结常规网络，只保留双目融合、offset和不确定性分支。"""
  prefixes = (
      'stereo_quality_encoder.', 'stereo_coarse_fusion.', 'stereo_fusion.',
      'stereo_attention.', 'target_context.', 'depth_offset.',
      'depth_log_variance.', 'depth_gate.')
  for name, parameter in model.named_parameters():
    parameter.requires_grad = name.startswith(prefixes)
  if hasattr(model, 'train_stereo_only'):
    model.train_stereo_only = True
  return [parameter for parameter in model.parameters() if parameter.requires_grad]


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  use_cuda = opt.gpus[0] >= 0 and torch.cuda.is_available()
  if not use_cuda:
    # 默认优先CUDA；机器没有CUDA时自动退回CPU。
    opt.gpus = [-1]
  opt.device = torch.device('cuda' if use_cuda else 'cpu')
  print('运行设备：{}'.format(
      torch.cuda.get_device_name(0) if use_cuda else 'CPU'))
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
  if opt.train_stereo_only:
    if opt.resume:
      raise ValueError('train_stereo_only不能与resume同时使用')
    trainable_parameters = configure_stereo_only_training(model)
    optimizer = torch.optim.Adam(trainable_parameters, opt.lr)
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print('仅训练双目offset分支：{:,}/{:,}个参数'.format(
        trainable_count, total_count))

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=opt.val_num_workers,
      pin_memory=opt.device.type == 'cuda'
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=opt.device.type == 'cuda',
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  validations_without_improvement = 0
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    log_dict_val = None
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      current_metric = log_dict_val[opt.metric]
      best, validations_without_improvement, improved = update_early_stopping(
          current_metric, best, opt.early_stopping_min_delta,
          validations_without_improvement)
      if improved:
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    # 最近权重每轮统一保存，避免验证分支的双层else产生歧义。
    save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
               epoch, model, optimizer)
    current_lr = optimizer.param_groups[0]['lr']
    logger.log_epoch(epoch, current_lr, log_dict_train, log_dict_val,
                     None if best == 1e10 else best)
    print('Epoch {}/{} 完成 | loss {:.4f} | lr {:.2e} | {}'.format(
        epoch, opt.num_epochs, log_dict_train['loss'], current_lr,
        os.path.join(opt.save_dir, 'results.png')))
    logger.write('\n')
    if (log_dict_val is not None and opt.early_stopping_patience > 0 and
        validations_without_improvement >= opt.early_stopping_patience):
      message = ('早停：连续{}次验证未改善，最佳{}为{:.4f}。\n'.format(
          validations_without_improvement, opt.metric, best))
      print(message.strip())
      logger.write(message)
      break
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
