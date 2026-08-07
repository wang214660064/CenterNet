#!/usr/bin/env python3
"""Export one KITTI annotation as a readable, self-explaining JSON file."""

import argparse
import json
import math
from pathlib import Path


OCCLUSION_NAMES = {
    0: "完全可见",
    1: "部分遮挡",
    2: "严重遮挡",
    3: "未知",
}


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Inspect one KITTI ground-truth annotation")
    parser.add_argument("--image-id", default="000008")
    parser.add_argument("--data-dir", type=Path, default=project_root / "data" / "kitti")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_calibration(path):
    calibration = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, values = line.split(":", 1)
        numbers = [float(value) for value in values.split()]
        columns = len(numbers) // 3 if len(numbers) in (9, 12) else 0
        calibration[key] = (
            [numbers[row * columns:(row + 1) * columns] for row in range(3)]
            if columns else numbers
        )
    return calibration


def read_objects(path):
    objects = []
    for object_id, line in enumerate(path.read_text().splitlines()):
        values = line.split()
        occluded = int(values[2])
        x1, y1, x2, y2 = [float(value) for value in values[4:8]]
        height, width, length = [float(value) for value in values[8:11]]
        x, y, z = [float(value) for value in values[11:14]]
        alpha = float(values[3])
        rotation_y = float(values[14])
        objects.append({
            "object_id": object_id,
            "class_name": values[0],
            "truncation_ratio": float(values[1]),
            "occlusion": {"level": occluded, "description": OCCLUSION_NAMES.get(occluded)},
            "observation_angle": {"radians": alpha, "degrees": math.degrees(alpha)},
            "bbox_2d_pixels": {
                "left": x1,
                "top": y1,
                "right": x2,
                "bottom": y2,
                "width": x2 - x1,
                "height": y2 - y1,
            },
            "dimensions_3d_meters": {"height": height, "width": width, "length": length},
            "location_camera_meters": {"x_right": x, "y_down": y, "z_forward_depth": z},
            "rotation_y": {"radians": rotation_y, "degrees": math.degrees(rotation_y)},
        })
    return objects


def main():
    args = parse_args()
    image_id = str(args.image_id).zfill(6)
    training = args.data_dir / "training"
    label_path = training / "label_2" / (image_id + ".txt")
    calibration_path = training / "calib" / (image_id + ".txt")
    image_path = training / "image_2" / (image_id + ".png")
    for path in (image_path, label_path, calibration_path):
        if not path.exists():
            raise FileNotFoundError(path)

    result = {
        "image_id": image_id,
        "image_path": str(image_path),
        "coordinate_system": {
            "x": "相机右方，单位：米",
            "y": "相机下方，单位：米",
            "z": "相机前方，即目标深度，单位：米",
            "3d_location_note": "KITTI的3D位置是目标底面的中心点",
        },
        "object_count": 0,
        "objects": read_objects(label_path),
        "camera_calibration": read_calibration(calibration_path),
    }
    result["object_count"] = len(result["objects"])
    output = args.output or args.data_dir / "annotations" / "readable" / (image_id + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("Exported {} objects to {}".format(result["object_count"], output))


if __name__ == "__main__":
    main()
