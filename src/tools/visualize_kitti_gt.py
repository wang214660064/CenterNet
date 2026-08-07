#!/usr/bin/env python3
"""Visualize KITTI 2D boxes and projected 3D boxes without OpenCV."""

from __future__ import print_function

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "Car": (255, 80, 70),
    "Van": (255, 145, 50),
    "Truck": (255, 210, 40),
    "Pedestrian": (70, 230, 100),
    "Person_sitting": (80, 200, 180),
    "Cyclist": (40, 180, 255),
    "Tram": (190, 100, 255),
    "Misc": (210, 210, 210),
    "DontCare": (110, 110, 110),
}

BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Visualize KITTI ground-truth boxes")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "kitti")
    parser.add_argument("--image-id", default="000010", help="Six-digit KITTI image id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-3d", action="store_true", help="Only draw 2D boxes")
    return parser.parse_args()


def load_projection(calib_path):
    for line in calib_path.read_text().splitlines():
        if line.startswith("P2:"):
            return np.asarray([float(v) for v in line.split()[1:]], dtype=np.float64).reshape(3, 4)
    raise ValueError("P2 is missing from {}".format(calib_path))


def compute_box_3d(dimensions, location, rotation_y):
    h, w, length = dimensions
    x_corners = np.array([length / 2, length / 2, -length / 2, -length / 2,
                          length / 2, length / 2, -length / 2, -length / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2,
                          w / 2, -w / 2, -w / 2, w / 2])
    c, s = math.cos(rotation_y), math.sin(rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    corners = rotation.dot(np.vstack((x_corners, y_corners, z_corners)))
    return corners + np.asarray(location, dtype=np.float64).reshape(3, 1)


def project(points_3d, projection):
    homogeneous = np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))
    points_2d = projection.dot(homogeneous)
    if np.any(points_2d[2] <= 0.1):
        return None
    points_2d[:2] /= points_2d[2:3]
    return points_2d[:2].T


def read_labels(label_path):
    objects = []
    for line in label_path.read_text().splitlines():
        values = line.split()
        objects.append({
            "class": values[0],
            "truncated": float(values[1]),
            "occluded": int(values[2]),
            "alpha": float(values[3]),
            "bbox": [float(v) for v in values[4:8]],
            "dimensions": [float(v) for v in values[8:11]],
            "location": [float(v) for v in values[11:14]],
            "rotation_y": float(values[14]),
        })
    return objects


def main():
    args = parse_args()
    image_id = str(args.image_id).zfill(6)
    output_path = args.output or Path(__file__).resolve().parents[2] / "exp" / ("kitti_gt_{}.jpg".format(image_id))
    base = args.data_dir / "training"
    image_path = base / "image_2" / (image_id + ".png")
    label_path = base / "label_2" / (image_id + ".txt")
    calib_path = base / "calib" / (image_id + ".txt")
    for path in (image_path, label_path, calib_path):
        if not path.exists():
            raise FileNotFoundError(path)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    projection = load_projection(calib_path)
    objects = read_labels(label_path)

    for obj in objects:
        class_name = obj["class"]
        color = COLORS.get(class_name, (255, 255, 255))
        x1, y1, x2, y2 = obj["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        if class_name != "DontCare":
            depth = obj["location"][2]
            text = "{}  z={:.1f}m".format(class_name, depth)
            text_box = draw.textbbox((x1, y1), text, font=font)
            text_height = text_box[3] - text_box[1] + 4
            draw.rectangle((x1, max(0, y1 - text_height), x1 + text_box[2] - text_box[0] + 4, y1), fill=color)
            draw.text((x1 + 2, max(0, y1 - text_height + 1)), text, fill=(0, 0, 0), font=font)

        if args.no_3d or class_name == "DontCare":
            continue
        corners = compute_box_3d(obj["dimensions"], obj["location"], obj["rotation_y"])
        corners_2d = project(corners, projection)
        if corners_2d is None:
            continue
        for start, end in BOX_EDGES:
            draw.line((tuple(corners_2d[start]), tuple(corners_2d[end])), fill=color, width=3)
        # Front face diagonal helps reveal heading direction.
        draw.line((tuple(corners_2d[0]), tuple(corners_2d[5])), fill=(255, 255, 255), width=2)
        draw.line((tuple(corners_2d[1]), tuple(corners_2d[4])), fill=(255, 255, 255), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    print("Rendered {} objects to {}".format(len(objects), output_path))


if __name__ == "__main__":
    main()
