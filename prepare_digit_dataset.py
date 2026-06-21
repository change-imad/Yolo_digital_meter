#!/usr/bin/env python3
"""
prepare_digit_dataset.py
========================
利用原始水表数据集中 class_id=2 的大框标签，裁剪出数显长条区域，
并将单字小框坐标重映射到裁剪后的子图上，生成全新的单字检测数据集。

输入: dataset/water meter.v7i.yolo26/  你的原始数据集路径
输出: dataset_digits/                  生成的用于单字检测的yolo数据集
"""

import os
import glob
import logging
import argparse
import shutil

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 原始 class_id -> 真实数字 (0-9) 的映射 ──
CLASS_MAP = {
    0: 0,   # '0'
    1: 1,   # '1'
    3: 2,   # '2'
    4: 3,   # '3'
    5: 4,   # '4'
    6: 5,   # '5'
    7: 6,   # '6'
    8: 7,   # '7'
    9: 8,   # '8'
    10: 9,  # '9'
}
BIG_BOX_CLASS_ID = 2


def parse_yolo_label(label_path):
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cid, xc, yc, w, h))
    return boxes


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    xmin = (xc - w / 2) * img_w
    ymin = (yc - h / 2) * img_h
    xmax = (xc + w / 2) * img_w
    ymax = (yc + h / 2) * img_h
    return xmin, ymin, xmax, ymax


def xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    xc = ((xmin + xmax) / 2) / img_w
    yc = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return xc, yc, w, h


def process_split(src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir, clean=False):
    if clean:
        for d in [dst_images_dir, dst_labels_dir]:
            if os.path.isdir(d):
                shutil.rmtree(d)
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(src_images_dir, "*.*")))
    ok = 0
    skip_no_bigbox = 0
    skip_read_fail = 0
    skip_exists = 0

    for img_path in tqdm(image_files, desc=os.path.basename(os.path.dirname(src_images_dir))):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(src_labels_dir, basename + ".txt")

        dst_img_path = os.path.join(dst_images_dir, basename + ".jpg")
        if not clean and os.path.exists(dst_img_path):
            skip_exists += 1
            continue

        # 💡 【核心修改 1】使用 np.fromfile 绕过 OpenCV 无法读取特殊/乱码字符路径的 Bug
        try:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None

        if img is None:
            skip_read_fail += 1
            continue
        img_h, img_w = img.shape[:2]

        if not os.path.exists(label_path):
            skip_no_bigbox += 1
            continue
        boxes = parse_yolo_label(label_path)

        big_box = None
        digit_boxes = []
        for b in boxes:
            if b[0] == BIG_BOX_CLASS_ID:
                big_box = b
            else:
                digit_boxes.append(b)

        if big_box is None:
            skip_no_bigbox += 1
            continue

        _, big_xc, big_yc, big_w, big_h = big_box
        bx1, by1, bx2, by2 = yolo_to_xyxy(big_xc, big_yc, big_w, big_h, img_w, img_h)
        bx1i = max(int(np.floor(bx1)), 0)
        by1i = max(int(np.floor(by1)), 0)
        bx2i = min(int(np.ceil(bx2)), img_w)
        by2i = min(int(np.ceil(by2)), img_h)

        crop = img[by1i:by2i, bx1i:bx2i]
        crop_h, crop_w = crop.shape[:2]
        if crop_w <= 0 or crop_h <= 0:
            skip_no_bigbox += 1
            continue

        # 💡 【核心修改 2】使用 cv2.imencode 保存图片，确保包含特殊字符的目的路径也能正常写入
        dst_out_path = os.path.join(dst_images_dir, basename + ".jpg")
        _, img_encode = cv2.imencode('.jpg', crop)
        img_encode.tofile(dst_out_path)

        new_labels = []
        for cid, xc, yc, w, h in digit_boxes:
            if cid not in CLASS_MAP:
                continue

            xmin, ymin, xmax, ymax = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
            xmin -= bx1i
            ymin -= by1i
            xmax -= bx1i
            ymax -= by1i

            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, crop_w)
            ymax = min(ymax, crop_h)

            if xmax <= xmin or ymax <= ymin:
                continue

            new_xc, new_yc, new_w, new_h = xyxy_to_yolo(
                xmin, ymin, xmax, ymax, crop_w, crop_h
            )
            new_xc = np.clip(new_xc, 0.0, 1.0)
            new_yc = np.clip(new_yc, 0.0, 1.0)
            new_w = np.clip(new_w, 0.001, 1.0)
            new_h = np.clip(new_h, 0.001, 1.0)

            new_cid = CLASS_MAP[cid]
            new_labels.append(f"{new_cid} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")

        with open(os.path.join(dst_labels_dir, basename + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(new_labels))
            if new_labels:
                f.write("\n")

        ok += 1

    return ok, skip_no_bigbox, skip_read_fail, skip_exists


def generate_data_yaml(output_dir):
    data = {
        "path": os.path.abspath(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 10,
        "names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        # 💡 【核心修改 3】删掉了这里面多余重复的内层局部 import yaml
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    log.info(f"data.yaml 已生成: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="裁剪数显区域，生成单字检测数据集")
    parser.add_argument("--src", type=str, default="dataset/water meter.v7i.yolo26", help="原始数据集根目录")
    parser.add_argument("--dst", type=str, default="dataset_digits", help="输出数据集根目录")
    parser.add_argument("--clean", action="store_true", help="清空目标目录后重新生成")
    args = parser.parse_args()

    splits = {"train": "train", "valid": "valid", "test": "test"}
    total = {"ok": 0, "skip": 0, "fail": 0, "exists": 0}

    for split_name, folder in splits.items():
        src_img = os.path.join(args.src, folder, "images")
        src_lbl = os.path.join(args.src, folder, "labels")
        dst_img = os.path.join(args.dst, folder, "images")
        dst_lbl = os.path.join(args.dst, folder, "labels")

        if not os.path.isdir(src_img):
            log.warning(f"跳过不存在的 split: {split_name}")
            continue

        log.info(f"处理 {split_name}...")
        ok, skip, fail, exists = process_split(src_img, src_lbl, dst_img, dst_lbl, clean=args.clean)
        total["ok"] += ok
        total["skip"] += skip
        total["fail"] += fail
        total["exists"] += exists
        log.info(f"  {split_name}: 成功={ok}, 无大框跳过={skip}, 读取失败={fail}, 已存在跳过={exists}")

    generate_data_yaml(args.dst)
    log.info(f"全部完成: 成功={total['ok']}, 跳过={total['skip']}, 失败={total['fail']}, 已存在跳过={total['exists']}")


if __name__ == "__main__":
    main()