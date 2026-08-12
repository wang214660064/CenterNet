"""生成YOLOv8风格双目CenterNet论文网络框架图。"""

from __future__ import annotations

import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'architecture' / '11_yolov8_style_framework.svg'
W, H = 2400, 1120
SVG = []

COLORS = {
    'bg': '#F3F4F6', 'text': '#0F172A', 'muted': '#475569', 'line': '#475569',
    'input': '#BAE6FD', 'input_bd': '#2563EB',
    'backbone': '#93C5FD', 'backbone_bd': '#1D4ED8',
    'head': '#FDE68A', 'head_bd': '#B45309',
    'stereo': '#BBF7D0', 'stereo_bd': '#15803D',
    'fusion': '#E9D5FF', 'fusion_bd': '#7E22CE',
    'output': '#A5F3FC', 'output_bd': '#0E7490',
}


def add(value):
  SVG.append(value)


def esc(value):
  return html.escape(str(value), quote=True)


def text(x, y, value, size=18, color=None, anchor='middle', weight='400'):
  color = color or COLORS['text']
  add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
      f'text-anchor="{anchor}" font-weight="{weight}" '
      'font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif">'
      f'{esc(value)}</text>')


def rect(x, y, w, h, fill, stroke, width=2, radius=12):
  add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def node(x, y, w, h, title, lines, kind, number=None, innovation=None):
  fill = COLORS[kind]
  stroke = COLORS[kind + '_bd']
  rect(x, y, w, h, fill, stroke, 2.5)
  if number is not None:
    add(f'<circle cx="{x + 22}" cy="{y + 22}" r="15" fill="{stroke}"/>')
    text(x + 22, y + 28, number, 15, '#FFFFFF', 'middle', '500')
  title_x = x + 48 if number is not None else x + w / 2
  title_anchor = 'start' if number is not None else 'middle'
  text(title_x, y + 29, title, 19, COLORS['text'], title_anchor, '500')
  for index, line in enumerate(lines):
    text(x + w / 2, y + 58 + index * 23, line, 15, COLORS['muted'])
  if innovation:
    badge_w = len(innovation) * 17 + 24
    rect(x + w - badge_w - 12, y + 10, badge_w, 27, stroke, stroke, 0, 13)
    text(x + w - badge_w / 2 - 12, y + 29, innovation, 14, '#FFFFFF', 'middle', '500')


def cube(cx, cy, w, h, title, shape, kind):
  fill = COLORS[kind]
  stroke = COLORS[kind + '_bd']
  x, y, depth = cx - w / 2, cy - h / 2, 15
  add(f'<polygon points="{x},{y} {x+depth},{y-9} {x+w+depth},{y-9} {x+w},{y}" '
      f'fill="{fill}" fill-opacity="0.75" stroke="{stroke}" stroke-width="2"/>')
  add(f'<polygon points="{x+w},{y} {x+w+depth},{y-9} {x+w+depth},{y+h-9} {x+w},{y+h}" '
      f'fill="{fill}" fill-opacity="0.55" stroke="{stroke}" stroke-width="2"/>')
  rect(x, y, w, h, fill, stroke, 2, 3)
  text(cx, cy - 3, title, 16, COLORS['text'], 'middle', '500')
  text(cx, cy + 19, shape, 13, COLORS['muted'])


def arrow(points, label=None, color=None, dashed=False):
  color = color or COLORS['line']
  path = 'M ' + ' L '.join(f'{x} {y}' for x, y in points)
  dash = ' stroke-dasharray="8 6"' if dashed else ''
  marker = 'arrow-green' if color == COLORS['stereo_bd'] else (
      'arrow-purple' if color == COLORS['fusion_bd'] else 'arrow')
  add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" '
      f'stroke-linejoin="round" marker-end="url(#{marker})"{dash}/>')
  if label:
    x, y = points[len(points) // 2]
    box_w = max(76, len(label) * 16 + 20)
    rect(x - box_w / 2, y - 22, box_w, 27, '#FFFFFF', '#FFFFFF', 0, 5)
    text(x, y - 3, label, 14, color, 'middle', '500')


def build():
  add('<?xml version="1.0" encoding="UTF-8"?>')
  add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
  add('<title id="title">园区低速车双目CenterNet三维检测网络</title>')
  add('<desc id="desc">左图进入DLA骨干，左右图和标定进入SGBM；质量感知融合、几何offset与投影中心解码共同输出二维和三维检测结果。</desc>')
  add('<defs>'
      '<marker id="arrow" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto"><path d="M0,0 L11,5.5 L0,11 z" fill="#475569"/></marker>'
      '<marker id="arrow-green" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto"><path d="M0,0 L11,5.5 L0,11 z" fill="#15803D"/></marker>'
      '<marker id="arrow-purple" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto"><path d="M0,0 L11,5.5 L0,11 z" fill="#7E22CE"/></marker>'
      '</defs>')
  add(f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>')

  text(W / 2, 48, 'CenterNet Stereo 3D Detection', 31, '#0B3B66', 'middle', '500')
  text(W / 2, 77, 'Geometry Offset v4 + Projected Center v5', 18, COLORS['muted'])
  add('<line x1="45" y1="98" x2="2355" y2="98" stroke="#CBD5E1" stroke-width="2"/>')

  # 输入：三项必要内容。
  cube(120, 200, 135, 76, 'Left RGB', '[3,384,1280]', 'input')
  cube(120, 340, 135, 76, 'Right RGB', '[3,384,1280]', 'input')
  node(45, 445, 165, 78, 'P2 / P3', ['逐帧标定'], 'input')

  # 图像分支。
  node(300, 150, 245, 105, 'DLA-34 + DLAUp', ['仅处理左图', '输出步长4'], 'backbone', '1')
  cube(680, 202, 170, 86, 'image_features', '[64,96,320]', 'backbone')
  arrow([(188, 200), (298, 200)], '左图', COLORS['backbone_bd'])
  arrow([(545, 202), (593, 202)], None, COLORS['backbone_bd'])

  node(820, 135, 360, 145, 'CenterNet Heads',
       ['2D：hm / wh / reg', '深度：z_direct', '3D：dim / rot', '中心：proj_center_offset'],
       'head', '2')
  arrow([(765, 202), (818, 202)], None, COLORS['backbone_bd'])

  # 双目分支与质量特征。
  node(300, 350, 245, 115, 'StereoSGBM',
       ['Left + Right + P2/P3', '视差 → 表面深度'], 'stereo', '3')
  node(620, 350, 275, 115, 'SGBM质量编码',
       ['深度 + 有效比例', '局部离散度 + 梯度'], 'stereo', '4')
  arrow([(188, 200), (245, 200), (245, 385), (298, 385)], None, COLORS['stereo_bd'])
  arrow([(188, 340), (245, 340), (245, 410), (298, 410)], None, COLORS['stereo_bd'])
  arrow([(210, 484), (265, 484), (265, 440), (298, 440)], None, COLORS['stereo_bd'])
  arrow([(545, 408), (618, 408)], 'z_sgbm', COLORS['stereo_bd'])

  # 创新一：双尺度、ECA和目标上下文。
  node(1000, 350, 330, 135, '质量感知特征融合',
       ['image_features + SGBM质量', '轻量双尺度 + ECA', 'hm/wh目标上下文 → target_features'],
       'fusion', '5', '创新①')
  arrow([(765, 245), (765, 315), (1040, 315), (1040, 348)], '图像特征', COLORS['backbone_bd'])
  arrow([(895, 408), (998, 408)], '质量特征', COLORS['stereo_bd'])
  arrow([(1000, 280), (1000, 318), (1190, 318), (1190, 348)], 'hm / wh', COLORS['head_bd'], True)

  # 创新二：表面到中心的可解释修正。
  node(1430, 275, 340, 170, 'Geometry Offset v4',
       ['geometry = f(dim, rot)（detach）', 'offset = quality × geometry_gate × geometry', '+ residual_offset', 'z_stereo = z_sgbm + offset'],
       'fusion', '6', '创新②')
  arrow([(1330, 408), (1428, 408)], 'target_features', COLORS['fusion_bd'])
  arrow([(895, 430), (1360, 430), (1360, 420), (1428, 420)], 'z_sgbm / quality', COLORS['stereo_bd'])
  arrow([(1180, 240), (1375, 240), (1375, 325), (1428, 325)], 'dim / rot', COLORS['head_bd'], True)

  # 门控融合：候选深度与安全回退。
  node(1430, 540, 340, 135, '深度门控融合',
       ['z_stereo ↔ z_direct', 'gate + uncertainty + 硬安全边界', '输出 z_final；不可信时回退z_direct'],
       'fusion', '7')
  arrow([(1600, 445), (1600, 538)], 'z_stereo', COLORS['fusion_bd'])
  arrow([(1100, 280), (1100, 610), (1428, 610)], 'z_direct', COLORS['head_bd'])
  arrow([(1240, 485), (1240, 650), (1428, 650)], 'target_features', COLORS['fusion_bd'])

  # 创新三：二维中心和三维投影中心分开使用。
  node(1900, 350, 360, 175, 'Projected Center v5',
       ['2D框：bbox center + wh', '3D投影中心：bbox center + proj offset', 'center_3d_proj + z_final + P2', '→ 相机坐标 [x,y,z]'],
       'output', '8', '创新③')
  arrow([(1770, 610), (1835, 610), (1835, 475), (1898, 475)], 'z_final', COLORS['fusion_bd'])
  arrow([(1180, 205), (1845, 205), (1845, 395), (1898, 395)], '2D / proj / dim / rot', COLORS['head_bd'])
  arrow([(210, 484), (250, 484), (250, 740), (1815, 740), (1815, 435), (1898, 435)], 'P2', COLORS['input_bd'])

  node(1900, 650, 360, 120, '最终输出',
       ['类别、置信度、2D框', '3D尺寸、朝向、相机坐标、深度质量状态'],
       'output', '9')
  arrow([(2080, 525), (2080, 648)], None, COLORS['output_bd'])

  # 图例与应用边界。
  rect(45, 850, 2310, 205, '#FFFFFF', '#CBD5E1', 2, 10)
  text(70, 884, '图例', 18, COLORS['text'], 'start', '500')
  legend = [('input', '输入'), ('backbone', '图像骨干'), ('head', '检测头'),
            ('stereo', '双目几何'), ('fusion', '融合/创新模块'), ('output', '解码输出')]
  x = 70
  for kind, label in legend:
    rect(x, 910, 28, 20, COLORS[kind], COLORS[kind + '_bd'], 2, 4)
    text(x + 40, 926, label, 15, COLORS['text'], 'start')
    x += 210
  text(70, 970, '实线：推理数据流；虚线：仅表示detach后的单向特征供给。训练标签与损失不进入推理主图。', 16, COLORS['muted'], 'start')
  text(70, 1003, '园区策略：0～30m核心3D检测，30～50m远距预警，50m以上仅保留2D观察。', 16, COLORS['muted'], 'start')
  text(70, 1036, 'project2000验证集：v5 Car Moderate 3D AP_R40 = 42.24。教学验证原型，不可直接作为车辆控制或唯一制动依据。', 16, '#B91C1C', 'start')

  add('</svg>')
  return '\n'.join(SVG)


def main():
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(build(), encoding='utf-8')
  print('SVG generated:', OUT)


if __name__ == '__main__':
  main()
