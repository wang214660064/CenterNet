from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('task', default='ctdet',
                             help='ctdet | ddd | stereo_ddd | multi_pose | exdet')
    self.parser.add_argument('--dataset', default='coco',
                             help='coco | kitti | coco_hp | pascal')
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--debug', type=int, default=0,
                             help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # useful when lunching training with ipython notebook
                                  '4: save all visualizations to disk')
    self.parser.add_argument('--demo', default='', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 

    # system
    self.parser.add_argument('--gpus', default='0',
                             help='默认自动使用CUDA，无CUDA时回退CPU；-1强制CPU')
    self.parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    self.parser.add_argument('--no_progress_bar', action='store_true',
                             help='关闭终端动态训练进度条')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                             help='visualization threshold.')
    self.parser.add_argument('--debugger_theme', default='white', 
                             choices=['white', 'black'])
    
    # model
    self.parser.add_argument('--arch', default='dla_34', 
                             help='model architecture. Currently tested'
                                  'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                  'dlav0_34 | dla_34 | hourglass')
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '64 for resnets and 256 for dla.')
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')

    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.')
    
    # train
    self.parser.add_argument('--lr', type=float, default=1.25e-4, 
                             help='learning rate for batch size 32.')
    self.parser.add_argument('--lr_step', type=str, default='90,120',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--num_epochs', type=int, default=140,
                             help='total training epochs.')
    self.parser.add_argument('--batch_size', type=int, default=32,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--val_num_workers', type=int, default=4,
                             help='验证集DataLoader进程数')
    self.parser.add_argument('--early_stopping_patience', type=int, default=0,
                             help='连续多少次验证未改善后停止，0表示关闭')
    self.parser.add_argument('--early_stopping_min_delta', type=float,
                             default=0.0,
                             help='验证指标至少改善多少才重置早停计数')
    self.parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')

    # test
    self.parser.add_argument('--flip_test', action='store_true',
                             help='flip data augmentation.')
    self.parser.add_argument('--test_scales', type=str, default='1',
                             help='multi scale test augmentation.')
    self.parser.add_argument('--nms', action='store_true',
                             help='run nms in testing.')
    self.parser.add_argument('--K', type=int, default=100,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--fix_res', action='store_true',
                             help='fix testing resolution or keep '
                                  'the original resolution')
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')

    # dataset
    self.parser.add_argument('--not_rand_crop', action='store_true',
                             help='not use the random crop data augmentation'
                                  'from CornerNet.')
    self.parser.add_argument('--shift', type=float, default=0.1,
                             help='when not using random crop'
                                  'apply shift augmentation.')
    self.parser.add_argument('--scale', type=float, default=0.4,
                             help='when not using random crop'
                                  'apply scale augmentation.')
    self.parser.add_argument('--rotate', type=float, default=0,
                             help='when not using random crop'
                                  'apply rotation augmentation.')
    self.parser.add_argument('--flip', type = float, default=0.5,
                             help='probability of applying flip augmentation.')
    self.parser.add_argument('--no_color_aug', action='store_true',
                             help='not use the color augmenation '
                                  'from CornerNet')
    # multi_pose
    self.parser.add_argument('--aug_rot', type=float, default=0, 
                             help='probability of applying '
                                  'rotation augmentation.')
    # ddd
    self.parser.add_argument('--aug_ddd', type=float, default=0.5,
                             help='probability of applying crop augmentation.')
    self.parser.add_argument('--rect_mask', action='store_true',
                             help='for ignored object, apply mask on the '
                                  'rectangular region or just center point.')
    self.parser.add_argument('--kitti_split', default='project2000',
                             help='different validation split for kitti: '
                                  'project2000 | 3dop | subcnn')

    # loss
    self.parser.add_argument('--mse_loss', action='store_true',
                             help='use mse loss or focal loss to train '
                                  'keypoint heatmaps.')
    # ctdet
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    # multi_pose
    self.parser.add_argument('--hp_weight', type=float, default=1,
                             help='loss weight for human pose offset.')
    self.parser.add_argument('--hm_hp_weight', type=float, default=1,
                             help='loss weight for human keypoint heatmap.')
    # ddd
    self.parser.add_argument('--dep_weight', type=float, default=1,
                             help='loss weight for depth.')
    self.parser.add_argument('--dim_weight', type=float, default=1,
                             help='loss weight for 3d bounding box size.')
    self.parser.add_argument('--dimension_aware_weight', type=float,
                             default=0.0,
                             help='按真实尺寸归一化的3D尺寸Smooth L1损失权重')
    self.parser.add_argument('--dimension_aware_beta', type=float,
                             default=0.1,
                             help='相对尺寸Smooth L1转折点')
    self.parser.add_argument('--rot_weight', type=float, default=1,
                             help='loss weight for orientation.')
    self.parser.add_argument('--peak_thresh', type=float, default=0.2)
    self.parser.add_argument('--depth_offset_weight', type=float, default=1.0,
                             help='SGBM深度offset不确定性损失权重')
    self.parser.add_argument('--depth_offset_loss', default='laplace',
                             choices=['laplace', 'huber'],
                             help='offset回归损失模式')
    self.parser.add_argument('--depth_offset_huber_delta', type=float, default=1.0,
                             help='Huber offset损失转折点，单位m')
    self.parser.add_argument('--depth_uncertainty_calibration_weight',
                             type=float, default=0.05,
                             help='Huber模式下不确定性校准损失权重')
    self.parser.add_argument('--depth_gate_weight', type=float, default=0.2,
                             help='可学习SGBM门控分类损失权重')
    self.parser.add_argument('--depth_fusion_weight', type=float, default=0.5,
                             help='融合后深度Huber损失权重')
    self.parser.add_argument('--proj_center_weight', type=float, default=1.0,
                             help='3D中心投影偏移Smooth L1损失权重')
    self.parser.add_argument('--proj_center_xy_weight', type=float, default=0.0,
                             help='投影中心相机坐标XY一致性损失权重')
    self.parser.add_argument('--proj_center_xy_beta', type=float, default=0.2,
                             help='相机坐标XY Smooth L1转折点，单位m')
    self.parser.add_argument('--proj_center_iou_weight', type=float,
                             default=0.0,
                             help='尺度归一化3D中心重叠代理损失权重')
    self.parser.add_argument('--proj_center_iou_beta', type=float, default=0.1,
                             help='中心完全错开时归一化Huber损失转折点')
    self.parser.add_argument('--proj_center_max_offset', type=float, default=64.0,
                             help='投影中心最大偏移，单位为输出特征格')
    self.parser.add_argument('--depth_gate_focal_gamma', type=float, default=2.0,
                             help='门控Focal Loss的gamma')
    self.parser.add_argument('--depth_gate_focal_alpha', type=float, default=0.5,
                             help='门控Focal Loss的正类alpha')
    self.parser.add_argument('--depth_gate_ambiguity_margin', type=float,
                             default=0.2,
                             help='两种深度误差差小于该值时忽略gate监督，单位m')
    self.parser.add_argument('--depth_gate_max_regret', type=float, default=4.0,
                             help='门控错误代价权重的最大附加值')
    self.parser.add_argument('--depth_gate_core_range_weight', type=float,
                             default=1.0,
                             help='15～30m Gate校准样本附加权重')
    self.parser.add_argument('--depth_gate_mid_quality_weight', type=float,
                             default=1.0,
                             help='中等SGBM质量Gate校准样本附加权重')
    self.parser.add_argument('--campus_near_distance', type=float, default=15.0,
                             help='园区近距离边界，单位m')
    self.parser.add_argument('--campus_core_distance', type=float, default=30.0,
                             help='园区核心3D检测边界，单位m')
    self.parser.add_argument('--campus_warning_distance', type=float, default=50.0,
                             help='园区远距离预警边界，单位m')
    self.parser.add_argument('--campus_near_weight', type=float, default=2.0,
                             help='0～15m双目损失权重')
    self.parser.add_argument('--campus_core_weight', type=float, default=1.5,
                             help='15～30m双目损失权重')
    self.parser.add_argument('--campus_warning_weight', type=float, default=0.5,
                             help='30～50m双目损失权重')
    self.parser.add_argument('--campus_beyond_weight', type=float, default=0.0,
                             help='50m以上双目损失权重')
    self.parser.add_argument('--stereo_num_disparities', type=int, default=128,
                             help='SGBM视差搜索范围，必须为16的倍数')
    self.parser.add_argument('--stereo_block_size', type=int, default=5,
                             help='SGBM匹配窗口，必须为正奇数')
    self.parser.add_argument('--stereo_max_depth', type=float, default=80.0,
                             help='参与offset训练的最大SGBM深度')
    self.parser.add_argument('--stereo_min_quality', type=float, default=0.5,
                             help='启用SGBM修正深度的最低质量')
    self.parser.add_argument('--stereo_far_distance', type=float, default=30.0,
                             help='开始使用严格SGBM门控的距离，单位m')
    self.parser.add_argument('--stereo_far_min_quality', type=float, default=0.8,
                             help='远距离启用SGBM修正的最低质量')
    self.parser.add_argument('--stereo_quality_window', type=int, default=31,
                             help='计算局部SGBM有效比例的窗口尺寸')
    self.parser.add_argument('--depth_offset_max_uncertainty', type=float, default=10.0,
                             help='启用offset修正的最大预测标准差')
    self.parser.add_argument('--depth_offset_far_max_uncertainty', type=float, default=3.0,
                             help='远距离启用offset修正的最大预测标准差')
    self.parser.add_argument('--depth_offset_max_abs', type=float, default=8.0,
                             help='offset修正绝对上限，单位m')
    self.parser.add_argument('--depth_offset_max_ratio', type=float, default=0.15,
                             help='offset修正相对SGBM深度的上限比例')
    self.parser.add_argument('--depth_offset_min_limit', type=float, default=2.0,
                             help='近距离允许的最小offset上限，单位m')
    self.parser.add_argument('--train_stereo_only', action='store_true',
                             help='冻结骨干和常规检测头，只训练双目offset分支')
    self.parser.add_argument('--train_gate_only', action='store_true',
                             help='只训练深度Gate头，用于单变量校准A/B')
    self.parser.add_argument('--train_projected_center_only', action='store_true',
                             help='只训练3D中心投影偏移头，用于单变量A/B')
    self.parser.add_argument('--train_dimension_only', action='store_true',
                             help='只训练3D尺寸头，用于Dimension v6b消融')
    
    # task
    # ctdet
    self.parser.add_argument('--norm_wh', action='store_true',
                             help='L1(pred / target, 1) or L1(pred, target)')
    self.parser.add_argument('--dense_wh', action='store_true',
                             help='apply weighted regression near center or '
                                  'just apply regression on center point.')
    self.parser.add_argument('--cat_spec_wh', action='store_true',
                             help='category specific bounding box size.')
    self.parser.add_argument('--not_reg_offset', action='store_true',
                             help='not regress local offset.')
    # exdet
    self.parser.add_argument('--agnostic_ex', action='store_true',
                             help='use category agnostic extreme points.')
    self.parser.add_argument('--scores_thresh', type=float, default=0.1,
                             help='threshold for extreme point heatmap.')
    self.parser.add_argument('--center_thresh', type=float, default=0.1,
                             help='threshold for centermap.')
    self.parser.add_argument('--aggr_weight', type=float, default=0.0,
                             help='edge aggregation weight.')
    # multi_pose
    self.parser.add_argument('--dense_hp', action='store_true',
                             help='apply weighted pose regression near center '
                                  'or just apply regression on center point.')
    self.parser.add_argument('--not_hm_hp', action='store_true',
                             help='not estimate human joint heatmap, '
                                  'directly use the joint offset from center.')
    self.parser.add_argument('--not_reg_hp_offset', action='store_true',
                             help='not regress local offset for '
                                  'human joint heatmaps.')
    self.parser.add_argument('--not_reg_bbox', action='store_true',
                             help='not regression bounding box size.')
    
    # ground truth validation
    self.parser.add_argument('--eval_oracle_hm', action='store_true', 
                             help='use ground center heatmap.')
    self.parser.add_argument('--eval_oracle_wh', action='store_true', 
                             help='use ground truth bounding box size.')
    self.parser.add_argument('--eval_oracle_offset', action='store_true', 
                             help='use ground truth local heatmap offset.')
    self.parser.add_argument('--eval_oracle_kps', action='store_true', 
                             help='use ground truth human pose offset.')
    self.parser.add_argument('--eval_oracle_hmhp', action='store_true', 
                             help='use ground truth human joint heatmaps.')
    self.parser.add_argument('--eval_oracle_hp_offset', action='store_true', 
                             help='use ground truth human joint local offset.')
    self.parser.add_argument('--eval_oracle_dep', action='store_true', 
                             help='use ground truth depth.')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.debug > 0:
      opt.num_workers = 0
      opt.batch_size = 1
      opt.gpus = [opt.gpus[0]]
      opt.master_batch_size = -1

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes

    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
    
    if opt.task == 'exdet':
      # assert opt.dataset in ['coco']
      num_hm = 1 if opt.agnostic_ex else opt.num_classes
      opt.heads = {'hm_t': num_hm, 'hm_l': num_hm, 
                   'hm_b': num_hm, 'hm_r': num_hm,
                   'hm_c': opt.num_classes}
      if opt.reg_offset:
        opt.heads.update({'reg_t': 2, 'reg_l': 2, 'reg_b': 2, 'reg_r': 2})
    elif opt.task in ['ddd', 'stereo_ddd']:
      # assert opt.dataset in ['gta', 'kitti', 'viper']
      opt.heads = {'hm': opt.num_classes, 'dep': 1, 'rot': 8, 'dim': 3}
      if opt.reg_bbox:
        opt.heads.update(
          {'wh': 2})
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
      if opt.task == 'stereo_ddd':
        opt.heads.update({
            'depth_offset': 1, 'depth_log_variance': 1, 'depth_gate': 1,
            'proj_center_offset': 2})
    elif opt.task == 'ctdet':
      # assert opt.dataset in ['pascal', 'coco']
      opt.heads = {'hm': opt.num_classes,
                   'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    elif opt.task == 'multi_pose':
      # assert opt.dataset in ['coco_hp']
      opt.flip_idx = dataset.flip_idx
      opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
      if opt.hm_hp:
        opt.heads.update({'hm_hp': 17})
      if opt.reg_hp_offset:
        opt.heads.update({'hp_offset': 2})
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

  def init(self, args=''):
    default_dataset_info = {
      'ctdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'exdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'multi_pose': {
        'default_resolution': [512, 512], 'num_classes': 1, 
        'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
        'dataset': 'coco_hp', 'num_joints': 17,
        'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
                     [11, 12], [13, 14], [15, 16]]},
      'ddd': {'default_resolution': [384, 1280], 'num_classes': 3, 
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'dataset': 'kitti'},
      'stereo_ddd': {'default_resolution': [384, 1280], 'num_classes': 3,
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'dataset': 'kitti'},
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt
