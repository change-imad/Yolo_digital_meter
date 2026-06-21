#!/usr/bin/env python3
"""
双 YOLO Pipeline 逐阶段可视化调试工具

流程（与 inference_pipeline.py 主 Pipeline 严格对齐）:
  阶段1: YOLO-OBB 检测数显大框 → 原图上画旋转框
  阶段2: 透视变换校正裁剪数显区域
  阶段2.5: CLAHE 增强预处理（可开关，默认开启）
  阶段3: YOLO 单字检测 (0-9) → 增强图上画每个数字框 + 类别标签
  阶段4: 按位置排序拼接读数结果

用法:
  python debug/debug_visualize_yolo.py --source dataset_obb/test/images/
  python debug/debug_visualize_yolo.py --source test.jpg --show
  python debug/debug_visualize_yolo.py --source test.jpg --no-enhance  # A/B 对比
"""

import os
import sys
import argparse
import logging

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path = [p for p in sys.path if p != "" and p != os.getcwd()]
sys.path.insert(0, ROOT)

from inference_pipeline import (
    YOLOBBDetector, get_corrected_screen, _perspective_crop, enhance_crop
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# 数字类别名
DIGIT_NAMES = [str(i) for i in range(10)]


# ── 绘图工具 ──

def draw_obb_box(img, corners, color=(0, 255, 0), thickness=2, label=None):
    """绘制 OBB 四边形框 + 可选标签。"""
    h, w = img.shape[:2]
    pts = np.array(corners, dtype=np.float32).reshape(4, 2)
    if pts.max() <= 1.5:
        pts[:, 0] *= w
        pts[:, 1] *= h
    pts_int = pts.astype(np.int32)
    cv2.polylines(img, [pts_int], isClosed=True, color=color, thickness=thickness)
    if label:
        cv2.putText(img, label, (pts_int[0][0], pts_int[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_digit_boxes(img, detections, conf_threshold=0.3):
    """
    在裁剪图上绘制所有单字检测框。
    detections: list of (class_id, x_c, y_c, w, h, conf) 归一化坐标
    返回: (vis_img, reading_str)
    """
    h, w = img.shape[:2]
    vis = img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # 过滤低置信度 + 按 x 中心排序
    valid = [(cid, xc, yc, bw, bh, conf)
             for cid, xc, yc, bw, bh, conf in detections
             if conf >= conf_threshold]
    valid.sort(key=lambda d: d[1])  # 按 x_c 从左到右

    reading_chars = []
    colors = [
        (255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128),
        (0, 200, 200), (200, 0, 200), (200, 200, 0), (100, 100, 255), (255, 100, 100),
    ]

    for cid, xc, yc, bw, bh, conf in valid:
        # 归一化 → 像素
        px = int(xc * w)
        py = int(yc * h)
        pw = int(bw * w)
        ph = int(bh * h)
        x1, y1 = px - pw // 2, py - ph // 2
        x2, y2 = x1 + pw, y1 + ph

        color = colors[cid % len(colors)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        label = f"{DIGIT_NAMES[cid]} {conf:.1f}"
        cv2.putText(vis, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        reading_chars.append(DIGIT_NAMES[cid])

    return vis, "".join(reading_chars)


# ── 检测逻辑 ──

def detect_digits(digit_model, crop_img, imgsz=416, conf=0.3):
    """
    用单字检测模型检测裁剪图中的数字。
    返回: list of (class_id, x_c, y_c, w, h, conf)

    与主 Pipeline 对齐：torch.no_grad() + .cpu().item() 立即释放显存引用。
    """
    # torch.no_grad() 避免隐式梯度记录占用显存
    with torch.no_grad():
        results = digit_model(crop_img, imgsz=imgsz, conf=conf, verbose=False)
    detections = []
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            # .cpu().item() 立即取值，释放 GPU 张量引用
            cls = int(boxes.cls[i].cpu().item())
            xc, yc, w, h = boxes.xywhn[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().item())
            detections.append((cls, xc, yc, w, h, conf))
    return detections


# ── 拼接调试图 ──

def make_debug_image(original, corners, crop_perspective, digit_vis, reading,
                     enhanced=False):
    """
    布局 (2列 × 2行):
      +---------------------------+----------------------------+
      |  原图 + OBB 大框           |  透视校正裁剪（原始）       |
      +---------------------------+----------------------------+
      |  增强图 + 单字检测框        |  读数结果                   |
      +---------------------------+----------------------------+
    """
    col_w, row_h = 500, 250

    def pad_to(cell, h, w):
        ch, cw = cell.shape[:2]
        canvas = np.full((h, w, 3), 20, dtype=np.uint8)
        y_off = max((h - ch) // 2, 0)
        x_off = max((w - cw) // 2, 0)
        # 如果 cell 比画布大，先缩放
        if ch > h or cw > w:
            s = min(w / cw, h / ch) * 0.95
            cell = cv2.resize(cell, (int(cw * s), int(ch * s)))
            ch, cw = cell.shape[:2]
            y_off = (h - ch) // 2
            x_off = (w - cw) // 2
        canvas[y_off:y_off+ch, x_off:x_off+cw] = cell
        return canvas

    # ── 左上: 原图 + OBB ──
    tl = original.copy()
    if corners is not None:
        draw_obb_box(tl, corners, label="screen")
    h1, w1 = tl.shape[:2]
    scale = min(col_w / w1, row_h / h1)
    tl = cv2.resize(tl, (int(w1 * scale), int(h1 * scale)))
    cv2.putText(tl, "1. OBB Detect", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # ── 右上: 透视校正裁剪（原始，便于与增强后对比）──
    if crop_perspective is not None:
        tr = crop_perspective.copy()
        if len(tr.shape) == 2:
            tr = cv2.cvtColor(tr, cv2.COLOR_GRAY2BGR)
    else:
        tr = np.zeros((40, 200, 3), dtype=np.uint8)
    cv2.putText(tr, "2. Perspective Crop", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

    # ── 左下: 增强后图 + 单字框（真实推理输入）──
    if digit_vis is not None:
        bl = digit_vis.copy()
    else:
        bl = np.zeros((40, 200, 3), dtype=np.uint8)
    bl_title = "3. Digit Detect (Enhanced)" if enhanced else "3. Digit Detect"
    cv2.putText(bl, bl_title, (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

    # ── 右下: 读数结果 ──
    br = np.full((row_h, col_w, 3), 30, dtype=np.uint8)
    cv2.putText(br, "4. Reading", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if reading:
        cv2.putText(br, reading, (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
    else:
        cv2.putText(br, "NO READING", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # ── 拼接 ──
    top = np.hstack([pad_to(tl, row_h, col_w), pad_to(tr, row_h, col_w)])
    bot = np.hstack([pad_to(bl, row_h, col_w), pad_to(br, row_h, col_w)])
    return np.vstack([top, bot])


# ── 主逻辑 ──


def process_single(obb_detector, digit_model, img_path, output_dir, args):
    img = cv2.imread(img_path)
    if img is None:
        log.warning(f"无法读取: {img_path}")
        return None

    basename = os.path.splitext(os.path.basename(img_path))[0]

    # 阶段1: OBB 检测大框
    detect_result = obb_detector.detect(img)
    corners = detect_result[0] if detect_result[0] is not None else None

    # 阶段2: 透视变换校正裁剪
    crop_perspective = None
    if corners is not None:
        crop_perspective = _perspective_crop(img, corners)

    # 阶段2.5: CLAHE 增强预处理（与主 Pipeline 对齐）
    # crop_enhanced 送入单字检测；crop_perspective 保留原始裁剪用于可视化对比
    crop_enhanced = crop_perspective
    if crop_perspective is not None:
        crop_enhanced = enhance_crop(crop_perspective, enable=args.enhance)

    # 阶段3: 单字检测（基于增强后图像，与主 Pipeline 推理输入一致）
    detections = []
    digit_vis = None
    reading = None
    if crop_enhanced is not None:
        detections = detect_digits(digit_model, crop_enhanced,
                                   imgsz=args.digit_imgsz, conf=args.digit_conf)
        # 检测框绘制在增强后图像上，反映真实推理输入
        digit_vis, reading = draw_digit_boxes(
            crop_enhanced, detections, conf_threshold=args.digit_conf)

    # 拼接调试图 (2×2 布局)
    # 右上: 原始裁剪；左下: 增强后裁剪 + 检测框（便于对比增强效果）
    debug_img = make_debug_image(img, corners, crop_perspective, digit_vis, reading,
                                 enhanced=args.enhance)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{basename}_debug.jpg")
    cv2.imwrite(out_path, debug_img)

    return {
        "file": basename,
        "obb_detected": corners is not None,
        "digit_count": len(detections),
        "reading": reading,
    }


def main():
    parser = argparse.ArgumentParser(description="双 YOLO Pipeline 可视化调试")
    parser.add_argument("--source", type=str, required=True,
                        help="图片路径或目录")
    # OBB 模型
    parser.add_argument("--obb_weights", type=str,
                        default=os.path.join(ROOT, "runs/obb/train_3/weights/best.pt"),
                        help="YOLO-OBB 屏幕定位模型")
    parser.add_argument("--obb_imgsz", type=int, default=640)
    parser.add_argument("--obb_conf", type=float, default=0.5)
    # 单字模型
    parser.add_argument("--digit_weights", type=str,
                        default=os.path.join(ROOT, "runs/digit/train_3/weights/best.pt"),
                        help="YOLO 单字检测模型")
    parser.add_argument("--digit_imgsz", type=int, default=416)
    parser.add_argument("--digit_conf", type=float, default=0.3)
    # 通用
    parser.add_argument("--output", type=str,
                        default=os.path.join(ROOT, "debug", "output_4"),
                        help="调试图输出目录")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--enhance", dest="enhance", action="store_true", default=True,
                        help="启用 CLAHE 增强预处理（默认开启，与主 Pipeline 一致）")
    parser.add_argument("--no-enhance", dest="enhance", action="store_false",
                        help="关闭 CLAHE 增强，便于 A/B 对比效果")
    args = parser.parse_args()

    # 加载双模型
    log.info("加载 OBB 屏幕定位模型...")
    obb_detector = YOLOBBDetector(weights=args.obb_weights, device=args.device,
                                  imgsz=args.obb_imgsz, conf=args.obb_conf)

    log.info("加载单字检测模型...")
    from ultralytics import YOLO
    digit_model = YOLO(args.digit_weights)

    # 收集图片
    if os.path.isdir(args.source):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted(os.path.join(args.source, f)
                       for f in os.listdir(args.source)
                       if f.lower().endswith(exts))
    else:
        files = [args.source]
    if args.limit > 0:
        files = files[:args.limit]

    log.info(f"共 {len(files)} 张图片待处理 (CLAHE 增强: {'ON' if args.enhance else 'OFF'})")

    # 处理
    stats = {"total": 0, "obb_ok": 0, "digit_ok": 0}
    for i, fpath in enumerate(files):
        r = process_single(obb_detector, digit_model, fpath, args.output, args)
        if r is None:
            continue
        stats["total"] += 1
        if r["obb_detected"]:
            stats["obb_ok"] += 1
        if r["digit_count"] > 0:
            stats["digit_ok"] += 1
        status = f"READ={r['reading']}" if r["reading"] else "FAIL"
        log.info(f"[{i+1}/{len(files)}] {r['file']}: {status}  "
                 f"(digits={r['digit_count']})")

        if args.show:
            debug_img = cv2.imread(os.path.join(args.output, f"{r['file']}_debug.jpg"))
            if debug_img is not None:
                cv2.imshow("Dual YOLO Debug", debug_img)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
    if args.show:
        cv2.destroyAllWindows()

    log.info(f"统计: 总计={stats['total']}, OBB检出={stats['obb_ok']}, "
             f"单字检出={stats['digit_ok']}")
    log.info(f"调试图已保存到: {args.output}")


if __name__ == "__main__":
    main()