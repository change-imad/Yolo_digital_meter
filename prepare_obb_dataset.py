#!/usr/bin/env python3
"""
prepare_obb_dataset.py
======================
从原始规范数据集中提取 class_id 对应的大框标签，
生成 YOLO OBB 格式的屏幕定位数据集。

将水平框 (xc, yc, w, h) 转换为 OBB 四角点 (x1 y1 x2 y2 x3 y3 x4 y4)，
图片原样复制（ symlink ），仅保留大框标签行并重映射 class_id 为 0。

输入: dataset/<your_dataset>/  (标准 YOLO 格式，含大框 class_id)
输出: dataset_obb/             (YOLO OBB 格式，nc=1, digital_screen)
"""

import os
import glob
import logging
import argparse
import shutil

import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def xywh_to_corners(xc, yc, w, h):
    """
    YOLO 归一化 (xc, yc, w, h) → OBB 四角点 (TL TR BR BL)，归一化坐标。
    水平框无旋转，角点即矩形的四个角。
    """
    x1 = xc - w / 2  # 左
    y1 = yc - h / 2  # 上
    x2 = xc + w / 2  # 右
    y2 = yc + h / 2  # 下
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def process_split(src_images, src_labels, dst_images, dst_labels, big_class_id):
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(src_images, "*.*")))
    ok = 0
    skip = 0

    for img_path in tqdm(image_files, desc=os.path.basename(os.path.dirname(src_images))):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(src_labels, basename + ".txt")

        # 找大框行
        big_lines = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    if cid == big_class_id:
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        corners = xywh_to_corners(xc, yc, w, h)
                        big_lines.append(f"0 " + " ".join(f"{v:.6f}" for v in corners))

        if not big_lines:
            skip += 1
            continue

        # symlink 图片（节省空间）
        dst_img = os.path.join(dst_images, basename + ".jpg")
        if not os.path.exists(dst_img):
            os.symlink(os.path.abspath(img_path), dst_img)

        # 写 OBB 标签
        with open(os.path.join(dst_labels, basename + ".txt"), "w") as f:
            f.write("\n".join(big_lines) + "\n")

        ok += 1

    return ok, skip


def generate_data_yaml(output_dir):
    data = {
        "path": os.path.abspath(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["digital_screen"],
    }
    path = os.path.join(output_dir, "data.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    log.info(f"data.yaml 已生成: {path}")


def main():
    parser = argparse.ArgumentParser(description="从原始数据集提取大框，生成 OBB 屏幕定位数据集")
    parser.add_argument("--src", type=str, default="dataset/water meter.v7i.yolo26",
                        help="原始数据集根目录")
    parser.add_argument("--dst", type=str, default="dataset_obb",
                        help="输出数据集根目录")
    parser.add_argument("--big_class_id", type=int, default=2,
                        help="原始数据集中大框的 class_id")
    args = parser.parse_args()

    splits = {"train": "train", "valid": "valid", "test": "test"}
    total_ok, total_skip = 0, 0

    for split_name, folder in splits.items():
        src_img = os.path.join(args.src, folder, "images")
        src_lbl = os.path.join(args.src, folder, "labels")
        dst_img = os.path.join(args.dst, folder, "images")
        dst_lbl = os.path.join(args.dst, folder, "labels")

        if not os.path.isdir(src_img):
            log.warning(f"跳过不存在的 split: {split_name}")
            continue

        log.info(f"处理 {split_name}...")
        ok, skip = process_split(src_img, src_lbl, dst_img, dst_lbl, args.big_class_id)
        total_ok += ok
        total_skip += skip
        log.info(f"  {split_name}: 成功={ok}, 无大框跳过={skip}")

    generate_data_yaml(args.dst)
    log.info(f"全部完成: 成功={total_ok}, 跳过={total_skip}")


if __name__ == "__main__":
    main()
