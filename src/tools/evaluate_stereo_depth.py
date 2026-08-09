#!/usr/bin/env python3
"""使用KITTI真值二维框批量评估SGBM深度。"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stereo_kitti_demo import (  # noqa: E402
    compute_disparity,
    disparity_to_depth,
    measure_detection,
    read_projection_matrices,
    stereo_parameters,
)


DEPTH_BUCKETS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80))


def parse_args():
    parser = argparse.ArgumentParser(description="批量评估KITTI双目目标深度")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "kitti")
    parser.add_argument("--split-file", type=Path,
                        default=PROJECT_ROOT / "data" / "kitti" / "ImageSets_3dop" / "val.txt")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "exp" / "stereo_stage2" / "sgbm_depth_metrics.json")
    parser.add_argument("--classes", default="Car,Pedestrian,Cyclist")
    parser.add_argument("--limit", type=int, default=0, help="0 表示评估整个划分")
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def read_labels(path, allowed_classes, max_depth):
    objects = []
    for line in path.read_text().splitlines():
        values = line.split()
        if values[0] not in allowed_classes:
            continue
        z = float(values[13])
        if z <= 0 or z > max_depth:
            continue
        objects.append({
            "class_name": values[0],
            "truncation": float(values[1]),
            "occlusion": int(values[2]),
            "bbox": [float(value) for value in values[4:8]],
            "label_z_m": z,
        })
    return objects


def depth_bucket(depth):
    for lower, upper in DEPTH_BUCKETS:
        if lower < depth <= upper:
            return "{}-{}m".format(lower, upper)
    return "其他"


def finite_or_none(value):
    return float(value) if math.isfinite(value) else None


def summarize(records):
    total = len(records)
    valid = [record for record in records if record["status"] == "valid"]
    if not valid:
        return {
            "target_count": total, "valid_count": 0, "failure_count": total,
            "valid_rate": 0.0, "mae_m": None, "median_ae_m": None,
            "rmse_m": None, "mean_relative_error": None, "signed_bias_m": None,
        }
    absolute_errors = np.asarray([record["absolute_error_m"] for record in valid], dtype=float)
    relative_errors = np.asarray([record["relative_error"] for record in valid], dtype=float)
    signed_errors = np.asarray([record["signed_error_m"] for record in valid], dtype=float)
    return {
        "target_count": total,
        "valid_count": len(valid),
        "failure_count": total - len(valid),
        "valid_rate": len(valid) / total if total else 0.0,
        "mae_m": finite_or_none(np.mean(absolute_errors)),
        "median_ae_m": finite_or_none(np.median(absolute_errors)),
        "rmse_m": finite_or_none(np.sqrt(np.mean(np.square(signed_errors)))),
        "mean_relative_error": finite_or_none(np.mean(relative_errors)),
        "signed_bias_m": finite_or_none(np.mean(signed_errors)),
    }


def grouped_summary(records, key):
    groups = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {name: summarize(items) for name, items in sorted(groups.items())}


def evaluate_frame(image_id, args, allowed_classes):
    training = args.data_dir / "training"
    left_path = training / "image_2" / (image_id + ".png")
    right_path = training / "image_3" / (image_id + ".png")
    calib_path = training / "calib" / (image_id + ".txt")
    label_path = training / "label_2" / (image_id + ".txt")
    for path in (left_path, right_path, calib_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    objects = read_labels(label_path, allowed_classes, args.max_depth)
    if not objects:
        return []
    left, right = cv2.imread(str(left_path)), cv2.imread(str(right_path))
    if left is None or right is None:
        raise RuntimeError("无法读取帧 {} 的双目图像".format(image_id))
    p2, p3 = read_projection_matrices(str(calib_path))
    params = stereo_parameters(p2, p3)
    disparity = compute_disparity(
        left, right, args.num_disparities, args.block_size)
    depth = disparity_to_depth(disparity, params, args.max_depth)

    records = []
    for target in objects:
        measured = measure_detection(target["bbox"], disparity, depth, params)
        record = dict(target)
        record["image_id"] = image_id
        record["depth_bucket"] = depth_bucket(target["label_z_m"])
        if measured is None:
            record.update({"status": "invalid_depth", "stereo_depth_m": None})
        else:
            estimate = measured["distance_m"]
            signed_error = estimate - target["label_z_m"]
            record.update({
                "status": "valid",
                "stereo_depth_m": estimate,
                "offset_target_m": target["label_z_m"] - estimate,
                "signed_error_m": signed_error,
                "absolute_error_m": abs(signed_error),
                "relative_error": abs(signed_error) / target["label_z_m"],
                "valid_depth_ratio": measured["valid_depth_ratio"],
                "depth_mad_m": measured["depth_mad_m"],
                "depth_iqr_m": measured["depth_iqr_m"],
            })
        records.append(record)
    return records


def main():
    args = parse_args()
    allowed_classes = {name.strip() for name in args.classes.split(",") if name.strip()}
    image_ids = [line.strip().zfill(6) for line in args.split_file.read_text().splitlines()
                 if line.strip()]
    if args.limit > 0:
        image_ids = image_ids[:args.limit]

    records = []
    failed_frames = []
    for index, image_id in enumerate(image_ids, start=1):
        try:
            records.extend(evaluate_frame(image_id, args, allowed_classes))
        except Exception as error:
            failed_frames.append({"image_id": image_id, "error": str(error)})
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(image_ids)):
            print("进度：{}/{} 帧，累计目标 {} 个".format(index, len(image_ids), len(records)))

    payload = {
        "evaluation_mode": "使用真值二维框评估双目深度，不包含2D检测误差",
        "backend": "sgbm",
        "split_file": str(args.split_file.resolve()),
        "frame_count": len(image_ids),
        "failed_frame_count": len(failed_frames),
        "classes": sorted(allowed_classes),
        "parameters": {
            "max_depth_m": args.max_depth,
            "num_disparities": args.num_disparities,
            "block_size": args.block_size,
        },
        "overall": summarize(records),
        "by_depth": grouped_summary(records, "depth_bucket"),
        "by_class": grouped_summary(records, "class_name"),
        "by_occlusion": grouped_summary(records, "occlusion"),
        "failed_frames": failed_frames,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("评估完成：{}".format(args.output.resolve()))
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
