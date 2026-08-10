#!/usr/bin/env python3
"""生成项目各版本的中文SVG架构图。"""

from __future__ import annotations

import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'readme' / 'architecture'
WIDTH, HEIGHT = 1600, 1000

COLORS = {
    'input': ('#EAF2FF', '#2563EB'),
    'network': ('#FFF4E5', '#D97706'),
    'geometry': ('#EAF8EF', '#15803D'),
    'fusion': ('#F2ECFF', '#7C3AED'),
    'output': ('#E8F7FA', '#0E7490'),
    'loss': ('#FFF0F0', '#DC2626'),
    'frozen': ('#F1F5F9', '#64748B'),
    'decision': ('#FFF8D9', '#A16207'),
}


def esc(value):
  return html.escape(str(value), quote=True)


def text_lines(x, y, lines, size=18, color='#334155', weight=400,
               anchor='start', line_height=27):
  spans = []
  for index, line in enumerate(lines):
    dy = 0 if index == 0 else line_height
    spans.append(
        '<tspan x="{}" dy="{}">{}</tspan>'.format(x, dy, esc(line)))
  return ('<text x="{}" y="{}" font-size="{}" font-weight="{}" '
          'fill="{}" text-anchor="{}">{}</text>').format(
              x, y, size, weight, color, anchor, ''.join(spans))


def node(x, y, width, height, title, lines, kind='network', badge=None):
  fill, stroke = COLORS[kind]
  result = [
      '<g>',
      '<rect x="{}" y="{}" width="{}" height="{}" rx="16" '
      'fill="{}" stroke="{}" stroke-width="2.5"/>'.format(
          x, y, width, height, fill, stroke),
      text_lines(x + 22, y + 38, [title], size=22, color='#0F172A', weight=700),
  ]
  if badge:
    badge_width = max(66, len(badge) * 17 + 22)
    result.extend([
        '<rect x="{}" y="{}" width="{}" height="30" rx="15" '
        'fill="{}"/>'.format(x + width - badge_width - 14, y + 13,
                             badge_width, stroke),
        text_lines(x + width - badge_width / 2 - 14, y + 34, [badge],
                   size=15, color='#FFFFFF', weight=700, anchor='middle'),
    ])
  if lines:
    result.append(text_lines(x + 22, y + 75, lines, size=17,
                             color='#334155', line_height=26))
  result.extend(['</g>'])
  return ''.join(result)


def arrow(x1, y1, x2, y2, label=None, dashed=False, color='#475569'):
  style = ' stroke-dasharray="9 7"' if dashed else ''
  marker = 'arrow-red' if color == '#DC2626' else 'arrow'
  parts = [
      '<path d="M {} {} L {} {}" fill="none" stroke="{}" '
      'stroke-width="3" marker-end="url(#{})"{}/>'.format(
          x1, y1, x2, y2, color, marker, style)
  ]
  if label:
    label_x, label_y = (x1 + x2) / 2, (y1 + y2) / 2 - 10
    parts.extend([
        '<rect x="{}" y="{}" width="{}" height="28" rx="7" '
        'fill="#FFFFFF" opacity="0.94"/>'.format(
            label_x - len(label) * 8.5 - 8, label_y - 20,
            len(label) * 17 + 16),
        text_lines(label_x, label_y, [label], size=15, color=color,
                   weight=600, anchor='middle'),
    ])
  return ''.join(parts)


def path_arrow(points, label=None, dashed=False, color='#475569'):
  d = 'M ' + ' L '.join('{} {}'.format(x, y) for x, y in points)
  style = ' stroke-dasharray="9 7"' if dashed else ''
  marker = 'arrow-red' if color == '#DC2626' else 'arrow'
  parts = [
      '<path d="{}" fill="none" stroke="{}" stroke-width="3" '
      'stroke-linejoin="round" marker-end="url(#{})"{}/>'.format(
          d, color, marker, style)
  ]
  if label:
    x, y = points[len(points) // 2]
    parts.append(text_lines(x + 10, y - 10, [label], size=15,
                            color=color, weight=600))
  return ''.join(parts)


def legend():
  items = [
      ('input', '输入/数据'), ('network', '神经网络'),
      ('geometry', '传统几何'), ('fusion', '融合模块'),
      ('output', '输出'), ('loss', '训练损失'), ('frozen', '冻结模块')]
  parts = []
  x = 60
  for kind, label in items:
    fill, stroke = COLORS[kind]
    parts.append('<rect x="{}" y="925" width="26" height="18" rx="4" '
                 'fill="{}" stroke="{}" stroke-width="2"/>'.format(
                     x, fill, stroke))
    parts.append(text_lines(x + 36, 940, [label], size=15, color='#475569'))
    x += 190
  return ''.join(parts)


def base_svg(title, subtitle, version, body, summary, filename):
  title_id = filename.replace('.', '-') + '-title'
  desc_id = filename.replace('.', '-') + '-desc'
  summary_lines = summary if isinstance(summary, list) else [summary]
  svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}"
  viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="{title_id} {desc_id}">
<title id="{title_id}">{esc(title)}</title>
<desc id="{desc_id}">{esc('；'.join(summary_lines))}</desc>
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6"
          orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="#475569"/>
  </marker>
  <marker id="arrow-red" markerWidth="12" markerHeight="12" refX="10" refY="6"
          orient="auto" markerUnits="strokeWidth">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="#DC2626"/>
  </marker>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">
    <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0F172A"
                  flood-opacity="0.10"/>
  </filter>
</defs>
<rect width="1600" height="1000" fill="#FFFFFF"/>
<rect x="0" y="0" width="1600" height="104" fill="#0F172A"/>
{text_lines(58, 55, [title], size=36, color='#FFFFFF', weight=700)}
{text_lines(60, 84, [subtitle], size=17, color='#CBD5E1')}
<rect x="1390" y="28" width="150" height="48" rx="24" fill="#2563EB"/>
{text_lines(1465, 60, [version], size=19, color='#FFFFFF', weight=700, anchor='middle')}
<g filter="url(#shadow)">{body}</g>
<rect x="48" y="800" width="1504" height="98" rx="16" fill="#F8FAFC"
      stroke="#CBD5E1" stroke-width="2"/>
{text_lines(74, 832, ['初学者一句话'], size=17, color='#2563EB', weight=700)}
{text_lines(74, 864, summary_lines, size=18, color='#0F172A', weight=600, line_height=25)}
{legend()}
</svg>'''
  (OUTPUT_DIR / filename).write_text(svg, encoding='utf-8')


def evolution_overview():
  cards = [
      (50, 160, '阶段1', ['2D检测 + SGBM测距', '无需训练3D模型'], 'input'),
      (420, 160, 'Stereo DDD', ['加入3D属性头', '学习深度offset'], 'network'),
      (790, 160, 'Fusion Gate v2', ['双尺度 + 质量编码', '学习融合gate'], 'fusion'),
      (1160, 160, 'Campus Gate v3', ['距离分层 + Focal', 'Moderate 3D AP 28.92'], 'decision'),
      (1160, 480, 'Geometry v4', ['尺寸/朝向几何先验', 'Moderate 3D AP 30.36'], 'geometry'),
      (790, 480, 'Projected v5', ['独立3D投影中心头', 'Moderate 3D AP 42.24'], 'network'),
      (420, 480, 'Projected v6a', ['像素 + 相机XY损失', 'Moderate 3D AP 42.29'], 'loss'),
      (50, 480, 'Projected v7', ['尺度归一化重叠损失', 'Moderate 3D AP 41.99'], 'loss'),
  ]
  body = []
  for x, y, title, lines, kind in cards:
    body.append(node(x, y, 320, 170, title, lines, kind))
  body.extend([
      arrow(370, 245, 410, 245), arrow(740, 245, 780, 245),
      arrow(1110, 245, 1150, 245),
      path_arrow([(1320, 335), (1320, 465)]),
      arrow(1160, 565, 1120, 565), arrow(790, 565, 750, 565),
      node(50, 690, 320, 100, '未进入主线的消融',
           ['LightStereo、3D全解冻、Dimension v6b', '均需独立评估，不替代主线'],
           'frozen', badge='旁支'),
      path_arrow([(420, 650), (360, 650)], '对照实验', dashed=True),
  ])
  base_svg(
      '园区低速车3D检测：版本演进总览',
      '每一版只解决一个主要问题，灰色旁支表示验证后没有进入主线',
      '总览', ''.join(body),
      ['主线从“能测距”逐步演进到“能学习深度质量、几何中心和相机坐标误差”。',
       '版本号是项目迭代号，不等同于CenterNet官方版本。'],
      '00_evolution_overview.svg')


def stage1():
  body = ''.join([
      node(55, 170, 275, 150, '左图 image_2',
           ['单张RGB图像', '用于2D目标检测'], 'input', '输入①'),
      node(55, 490, 275, 170, '双目与标定',
           ['image_2 + image_3', '逐帧 P2 / P3'], 'input', '输入②'),
      node(420, 170, 310, 150, 'CenterNet COCO',
           ['DLA-34预训练权重', '类别 + 置信度 + 2D框'], 'network'),
      node(420, 490, 310, 170, 'StereoSGBM',
           ['计算视差', '由 fx×baseline/视差 得深度'], 'geometry'),
      node(820, 315, 330, 190, '目标框内稳健测距',
           ['取框内中下区域', '中位数 + MAD过滤', '保留有效深度比例'], 'geometry'),
      node(1240, 270, 300, 280, '可观察输出',
           ['检测叠加图', '彩色视差图', 'JSON：框/距离/坐标', '风险颜色提示'],
           'output'),
      arrow(330, 245, 410, 245), arrow(330, 575, 410, 575),
      path_arrow([(730, 245), (775, 245), (775, 385), (810, 385)], '2D框'),
      path_arrow([(730, 575), (775, 575), (775, 435), (810, 435)], '深度图'),
      arrow(1150, 410, 1230, 410),
  ])
  base_svg(
      '阶段1：CenterNet 2D检测 + SGBM双目测距',
      '先把检测、几何测距、可视化和JSON输出跑通，不训练新的3D网络',
      '阶段1', body,
      ['CenterNet负责“哪里有车”，SGBM负责“车离多远”，最后在2D框里做稳健取值。',
       '限制：SGBM更接近可见表面，并不等于3D框中心深度。'],
      '01_stage1_centernet_sgbm.svg')


def stereo_baseline():
  body = ''.join([
      node(50, 170, 260, 150, '左图RGB', ['384×1280', '训练主图像'], 'input'),
      node(50, 500, 260, 150, 'SGBM输入', ['左右图 + P2/P3', '深度图 + 质量图'], 'input'),
      node(390, 150, 320, 190, 'DLA-34图像骨干',
           ['输出1/4特征图', 'hm / wh / reg', 'dep / dim / rot'], 'network'),
      node(390, 480, 320, 190, '早期双目融合分支',
           ['图像特征 + SGBM', 'depth_offset', 'depth_log_variance'], 'fusion'),
      node(800, 315, 330, 210, '深度修正与回退',
           ['z_stereo = z_sgbm + offset', '质量差：回退direct dep',
            '不确定性描述可信程度'], 'decision'),
      node(1220, 270, 320, 300, '3D检测输出',
           ['类别与2D框', '最终深度 z', '尺寸 h/w/l', '朝向 yaw',
            '相机坐标 x/y/z'], 'output'),
      arrow(310, 245, 380, 245), arrow(310, 575, 380, 575),
      path_arrow([(710, 245), (760, 245), (760, 365), (790, 365)], 'direct dep'),
      path_arrow([(710, 575), (760, 575), (760, 475), (790, 475)], 'offset'),
      arrow(1130, 420, 1210, 420),
  ])
  base_svg(
      'Stereo DDD基线：把SGBM接入可训练3D检测头',
      '从“外挂测距”升级为“网络学习SGBM表面深度到3D中心深度的修正”',
      '基线', body,
      ['图像网络预测3D属性，双目分支只修正深度；SGBM失败时仍有direct dep兜底。',
       '新增训练目标：depth_offset与depth_log_variance。'],
      '02_stereo_ddd_baseline.svg')


def fusion_gate_v2():
  body = ''.join([
      node(40, 155, 270, 170, '图像特征',
           ['DLA-34输出1/4特征', '同时构造1/8粗尺度'], 'network'),
      node(40, 500, 270, 180, 'SGBM质量编码',
           ['深度', '有效比例', '局部离散度', '深度梯度'], 'geometry'),
      node(380, 285, 290, 220, '双尺度融合',
           ['1/4：保留小目标细节', '1/8：扩大上下文', '融合后回到1/4尺度'], 'fusion'),
      node(750, 285, 260, 220, '特征增强',
           ['ECA通道注意力', '3×3 / 7×7 / 15×15', '目标级上下文聚合'], 'fusion'),
      node(1090, 155, 245, 170, 'offset',
           ['修正SGBM表面深度', 'Huber回归'], 'network'),
      node(1090, 355, 245, 150, 'uncertainty',
           ['估计修正可信度', '低质量时增大'], 'network'),
      node(1090, 535, 245, 150, 'learned gate',
           ['0：direct dep', '1：SGBM修正'], 'decision'),
      node(1400, 300, 160, 230, '连续融合',
           ['z_final', '= gate×', 'z_stereo', '+ (1-gate)×', 'z_direct'], 'output'),
      arrow(310, 240, 370, 350), arrow(310, 590, 370, 440),
      arrow(670, 395, 740, 395),
      path_arrow([(1010, 350), (1050, 350), (1050, 240), (1080, 240)]),
      arrow(1010, 395, 1080, 430),
      path_arrow([(1010, 440), (1050, 440), (1050, 610), (1080, 610)]),
      path_arrow([(1335, 240), (1370, 240), (1370, 370), (1390, 370)]),
      arrow(1335, 430, 1390, 415), arrow(1335, 610, 1390, 470),
  ])
  base_svg(
      'Fusion Gate v2：让网络判断SGBM什么时候可信',
      '双尺度、质量编码、注意力、目标级上下文和可学习gate首次形成完整闭环',
      'v2', body,
      ['不再用固定规则简单选择深度，而是让gate连续融合SGBM修正深度与direct dep。',
       '硬安全边界仍保留，学习gate不能绕过无效视差检查。'],
      '03_fusion_gate_v2.svg')


def campus_gate_v3():
  body = ''.join([
      node(50, 170, 300, 190, 'Fusion Gate v2主体',
           ['双尺度融合', '质量编码 + ECA', 'offset / uncertainty / gate'],
           'frozen', '继承'),
      node(50, 500, 300, 190, '两种深度候选',
           ['z_stereo_corrected', 'z_direct', '比较各自GT误差'], 'fusion'),
      node(430, 150, 300, 150, '0～15m',
           ['核心近距', '训练权重 2.0'], 'decision'),
      node(430, 325, 300, 150, '15～30m',
           ['核心规划距离', '训练权重 1.5'], 'decision'),
      node(430, 500, 300, 150, '30～50m',
           ['远距预警', '训练权重 0.5'], 'decision'),
      node(800, 230, 330, 260, '代价加权Focal Gate',
           ['错误选择代价越大 → 权重越高', '误差差 < 0.2m → 忽略',
            'gamma=2，最大regret=4'], 'loss'),
      node(800, 545, 330, 150, '50m以上',
           ['gate强制为0', '只保留2D观察用途'], 'frozen'),
      node(1210, 275, 330, 270, '园区安全输出',
           ['0～30m：核心3D检测', '30～50m：粗深度预警',
            '质量/不确定性不满足', '→ 回退direct dep'], 'output'),
      path_arrow([(350, 250), (390, 250), (390, 225), (420, 225)]),
      arrow(350, 595, 420, 400),
      arrow(730, 225, 790, 300), arrow(730, 400, 790, 360),
      arrow(730, 575, 790, 420),
      arrow(1130, 360, 1200, 390), arrow(1130, 620, 1200, 500),
  ])
  base_svg(
      'Campus Gate v3：按照园区距离分层训练',
      '把近距离安全目标放在最高优先级，远距离样本不再主导gate学习',
      'v3', body,
      ['同一个gate在不同距离使用不同训练权重，并重点惩罚“选错深度候选”的样本。',
       '正式结果：Car Moderate 3D AP_R40 = 28.92。'],
      '04_campus_gate_v3.svg')


def geometry_v4():
  body = ''.join([
      node(45, 160, 280, 170, 'SGBM目标深度',
           ['更接近可见表面', '附带局部质量quality'], 'geometry'),
      node(45, 500, 280, 180, '3D属性头',
           ['尺寸 dim：h/w/l', '朝向 rot：alpha', '输入前执行detach'],
           'frozen', '冻结'),
      node(395, 450, 355, 230, '几何表面→中心先验',
           ['geometry_offset = 0.5×', '(length×|cosα|', '+ width×|sinα|)',
            '再乘quality与学习门控'], 'geometry'),
      node(395, 160, 355, 180, '学习残差分支',
           ['图像 + SGBM融合特征', '修正车窗/侧面/遮挡', '输出residual_offset'],
           'network'),
      node(830, 280, 350, 240, 'Geometry Offset组合',
           ['depth_offset =', 'quality×geometry_gate×先验',
            '+ learned_residual', '仍受绝对/比例限幅'], 'fusion'),
      node(1260, 280, 290, 240, '最终深度',
           ['z_stereo = z_sgbm', '+ depth_offset', '再由gate与direct dep融合'],
           'output'),
      arrow(325, 245, 385, 245), arrow(325, 590, 385, 565),
      arrow(750, 250, 820, 350), arrow(750, 565, 820, 450),
      path_arrow([(325, 245), (790, 245), (790, 400), (820, 400)], 'z_sgbm'),
      arrow(1180, 400, 1250, 400),
  ])
  base_svg(
      'Geometry Offset v4：用尺寸和朝向解释表面到中心的距离',
      '把offset拆成“可解释几何先验 + 网络学习残差”，并阻断对dim/rot的反向影响',
      'v4', body,
      ['SGBM看见的是车辆表面；v4根据车辆长宽和观察角估计表面到3D中心的距离。',
       '正式结果：Car Moderate 3D AP_R40 = 30.36。'],
      '05_geometry_offset_v4.svg')


def projected_v5():
  body = ''.join([
      node(45, 150, 285, 170, 'DLA图像特征',
           ['原2D检测头保持不变', 'v4深度链路保持冻结'], 'frozen', '冻结v4'),
      node(45, 500, 285, 180, '监督投影中心',
           ['KITTI 3D底面中心', '上移 height/2', '通过当帧P2投影'],
           'input'),
      node(410, 260, 340, 230, 'proj_center_offset头',
           ['2通道：dx / dy', '目标 = 3D投影中心', '− 2D框中心',
            '仅训练148,226个参数'], 'network', '新增'),
      node(840, 150, 300, 190, '2D路径',
           ['2D中心 + wh', '恢复二维检测框', '不使用proj offset'], 'output'),
      node(840, 480, 300, 190, '3D中心路径',
           ['2D中心 + proj offset', '+ z_final + P2', '恢复相机x/y/z'], 'fusion'),
      node(1230, 275, 320, 270, '最终预测',
           ['同一目标保留两种中心', '2D框不被3D中心带偏', '3D框中心更加准确',
            '边缘大偏移做限幅'], 'output'),
      arrow(330, 235, 400, 325), arrow(330, 590, 400, 420),
      path_arrow([(330, 235), (790, 235), (830, 235)], '原2D中心'),
      arrow(750, 375, 830, 565),
      path_arrow([(1140, 245), (1180, 245), (1180, 350), (1220, 350)]),
      path_arrow([(1140, 575), (1180, 575), (1180, 470), (1220, 470)]),
  ])
  base_svg(
      'Projected Center v5：分开“2D框中心”和“3D几何中心投影点”',
      '修复单目3D检测中最关键的中心错位问题，2D框路径完全不受影响',
      'v5', body,
      ['2D框仍围绕可见框中心绘制；只有恢复3D相机坐标时才使用新的投影中心。',
       '正式结果：Car Moderate 3D AP_R40从30.36提升到42.24。'],
      '06_projected_center_v5.svg')


def projected_v6a():
  body = ''.join([
      node(45, 160, 290, 180, 'v5投影中心头',
           ['输入：DLA图像特征', '输出：dx / dy', '推理结构完全不变'],
           'network', '唯一可训练'),
      node(45, 500, 290, 170, '冻结的v5其他模块',
           ['v4深度/gate/offset', 'hm/wh/reg/dim/rot', '骨干与BatchNorm'],
           'frozen', '无梯度'),
      node(410, 150, 335, 200, '损失①：像素偏移',
           ['预测proj offset', '↔ GT proj offset', '距离加权Smooth L1'],
           'loss'),
      node(410, 470, 335, 230, '反投影所需信息',
           ['center_image = inverse_affine', '当帧P2标定矩阵',
            'z_final先执行detach'], 'geometry'),
      node(825, 300, 335, 240, '损失②：相机XY一致性',
           ['预测投影中心 + z_final + P2', '→ 预测相机x/y',
            '↔ GT相机x/y', 'Smooth L1，beta=0.2m'], 'loss', 'v6a新增'),
      node(1240, 300, 310, 240, '训练效果边界',
           ['梯度只回到proj head', '深度只提供数值，不更新',
            '同时兼顾像素与米制误差', '推理解码与v5相同'], 'output'),
      arrow(335, 245, 400, 245), arrow(335, 585, 400, 585),
      path_arrow([(335, 300), (370, 300), (370, 410), (815, 410)],
                 '预测中心'),
      arrow(745, 585, 815, 470), arrow(745, 250, 815, 350),
      arrow(1160, 420, 1230, 420),
      path_arrow([(745, 250), (1190, 250), (1190, 370), (1230, 370)]),
      path_arrow([(825, 500), (780, 500), (780, 735), (20, 735),
                  (20, 250), (60, 250)],
                 '梯度只回proj head', dashed=True, color='#DC2626'),
      arrow(410, 565, 380, 565),
  ])
  base_svg(
      'Projected Center v6a：同时优化像素误差和真实相机XY误差',
      '用最终深度衡量中心偏移带来的米制误差，但通过detach保护已经验证的深度链路',
      'v6a', body,
      ['像素损失保证投影点正确，XY损失保证最终相机坐标正确；两者共同训练同一个小头。',
       '400帧评估与v5基本持平，米制XY损失暂未带来稳定收益。'],
      '07_projected_center_v6a.svg')


def projected_v7_iou():
  body = ''.join([
      node(45, 150, 300, 185, 'v5投影中心头',
           ['输出proj dx/dy', '其余v5模块全部冻结', '只训练148,226个参数'],
           'network', '唯一可训练'),
      node(45, 500, 300, 190, '训练真值几何',
           ['KITTI尺寸[h,w,l]', 'rotation_y朝向', '生成相机X有效宽度和高度'],
           'input'),
      node(420, 145, 335, 210, '保留v5像素损失',
           ['预测proj offset', '↔ GT proj offset', '距离加权Smooth L1'],
           'loss'),
      node(420, 475, 335, 230, '反投影到相机坐标',
           ['预测中心 + detach(z_final)', '+ inverse_affine + 当帧P2',
            '得到预测x/y与真值x/y'], 'geometry'),
      node(830, 245, 345, 310, '尺度归一化重叠代理',
           ['横向误差 ÷ 朝向后车辆宽度', '纵向误差 ÷ 车辆高度',
            '计算中心重叠率', '无重叠时由Huber保留梯度',
            '权重0.2'], 'loss', 'v7新增'),
      node(1250, 280, 300, 245, '训练边界',
           ['v6a XY损失关闭', '深度/尺寸/朝向均detach或冻结',
            '推理解码仍与v5相同', '训练后必须做400帧A/B'],
           'output'),
      arrow(345, 240, 410, 240), arrow(345, 595, 410, 595),
      path_arrow([(345, 290), (380, 290), (380, 520), (410, 520)],
                 '预测中心'),
      arrow(755, 585, 820, 455), arrow(755, 250, 820, 335),
      arrow(1175, 400, 1240, 400),
      path_arrow([(830, 500), (790, 500), (790, 750), (20, 750),
                  (20, 230), (55, 230)],
                 '梯度只回proj head', dashed=True, color='#DC2626'),
  ])
  base_svg(
      'Projected Center v7：用车辆尺度归一化中心重叠误差',
      '比纯米制XY平均误差更接近3D IoU：小车辆对同样偏移更敏感，大车辆容忍度更高',
      'v7', body,
      ['v7从v5最佳权重开始，保留像素损失并加入0.2权重的可导重叠代理损失。',
       '400帧评估低于v5，尺度重叠代理暂未带来稳定收益。'],
      '08_projected_center_v7_iou.svg')


def dimension_v6b():
  body = ''.join([
      node(45, 170, 300, 190, 'v5完整模型',
           ['骨干、2D、深度、中心、朝向', '全部冻结并保持eval', '不改变推理解码'],
           'frozen', '冻结'),
      node(45, 500, 300, 180, 'KITTI尺寸真值',
           ['目标尺寸：[height, width, length]', '每一维独立作为监督', '只在训练阶段读取'],
           'input'),
      node(420, 250, 335, 260, 'dim尺寸头',
           ['输出3通道：h / w / l', '加载v5已有权重', '只训练该头4个参数张量', '不增加模型体量'],
           'network', '唯一可训练'),
      node(830, 220, 350, 310, 'Dimension-aware Smooth L1',
           ['relative_error = (pred − gt) / gt', '高度、宽度、长度按相对误差',
            '避免车长数值更大而主导损失', 'beta = 0.1', '沿用园区距离权重'],
           'loss', 'v6b新增'),
      node(1250, 265, 300, 255, '训练边界',
           ['原始dim L1权重设为0', '只启用相对尺寸损失',
            '深度/中心/朝向无梯度', '训练后对比尺寸MAE与3D AP'],
           'output'),
      arrow(345, 265, 410, 345), arrow(345, 590, 410, 430),
      arrow(755, 380, 820, 380), arrow(1180, 380, 1240, 380),
      path_arrow([(830, 485), (790, 485), (790, 750), (20, 750),
                  (20, 265), (55, 265)],
                 '梯度只回dim头', dashed=True, color='#DC2626'),
  ])
  base_svg(
      'Dimension v6b：让三维尺寸按相对误差公平学习',
      '只训练已有dim头，车长、高度和宽度不再因为绝对米制不同而获得不均衡的损失权重',
      'v6b', body,
      ['v6b从v5最佳权重开始，只更新尺寸头；不影响检测、深度、投影中心或朝向。',
       '当前状态：代码完成待训练，结果以project2000的400帧A/B评估为准。'],
      '09_dimension_v6b.svg')


def main():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  evolution_overview()
  stage1()
  stereo_baseline()
  fusion_gate_v2()
  campus_gate_v3()
  geometry_v4()
  projected_v5()
  projected_v6a()
  projected_v7_iou()
  dimension_v6b()
  print('已生成10张SVG：{}'.format(OUTPUT_DIR))


if __name__ == '__main__':
  main()
