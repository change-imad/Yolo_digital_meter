#!/usr/bin/env python3
"""
数字仪表读数推理 Pipeline (双 YOLO 版)

流程: 拉普拉斯模糊过滤 → YOLO-OBB 屏幕定位 → 裁剪数显区域
      → YOLO 单字检测(0-9) → 按位置排序拼接读数 → 时序投票 → 结果上报

支持 x86_64 (PC) 和 aarch64 (Jetson) 双平台。
"""

import os
import json
import csv
import logging
import platform
from collections import deque
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

log = logging.getLogger(__name__)

DIGIT_NAMES = [str(i) for i in range(10)]


# 1. 拉普拉斯清晰度判定（前端过滤）

def is_frame_blurry(frame: np.ndarray, threshold: float = 100.0) -> bool:
    """拉普拉斯方差法，True=模糊应丢弃。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


# 2. 透视变换裁剪辅助函数

def _perspective_crop(frame, corners):
    """
    用 OBB 角点做透视变换校正，自适应输出尺寸。
    返回正向的数显区域像素图，或在失败时回退到外接矩形裁剪。
    """
    h, w = frame.shape[:2]
    pts = np.array(corners, dtype=np.float32).reshape(4, 2)
    if pts.max() <= 1.5:
        pts[:, 0] *= w
        pts[:, 1] *= h

    # 根据 OBB 四条边的平均长度计算自适应输出尺寸
    def _dist(p1, p2):
        return np.linalg.norm(p1 - p2)
    width_top = _dist(pts[0], pts[1])
    width_bottom = _dist(pts[3], pts[2])
    height_left = _dist(pts[0], pts[3])
    height_right = _dist(pts[1], pts[2])
    out_w = max(int((width_top + width_bottom) / 2), 50)
    out_h = max(int((height_left + height_right) / 2), 20)

    return get_corrected_screen(frame, corners, output_size=(out_w, out_h))


# 3. YOLO-OBB 屏幕定位器

class ScreenDetector:
    """阶段一：YOLO-OBB 检测数显大框。"""

    def __init__(self, weights: str, device: str = "0",
                 imgsz: int = 640, conf: float = 0.5):
        from ultralytics import YOLO
        log.info(f"加载屏幕定位模型: {weights}")
        self.model = YOLO(weights, task="obb")
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def detect(self, frame: np.ndarray):
        """
        返回 (corners, crop_img) 或 (None, None)。
        corners: (4,2) 归一化角点
        crop_img: 透视变换校正后的数显区域像素图
        """
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf,
                             device=self.device, verbose=False)
        if not results or len(results[0].obb) == 0:
            return None, None

        obb = results[0].obb
        best_idx = obb.conf.argmax().item()
        box = obb.xywhr[best_idx].cpu().numpy()
        corners = self._xywhr_to_corners(box, frame.shape[:2])

        # 透视变换校正（自适应输出尺寸）
        crop = _perspective_crop(frame, corners)

        return corners, crop

    @staticmethod
    def _xywhr_to_corners(xywhr, img_shape):
        cx, cy, bw, bh, r = xywhr
        H, W = img_shape
        cx, cy = cx * W, cy * H
        bw, bh = bw * W, bh * H
        cos_a, sin_a = np.cos(r), np.sin(r)
        dx, dy = bw / 2, bh / 2
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = corners @ rot.T
        corners[:, 0] += cx
        corners[:, 1] += cy
        corners[:, 0] /= W
        corners[:, 1] /= H
        return corners


# 3. YOLO 单字检测器

class DigitDetector:
    """阶段二：检测裁剪图中的单个数字 0-9。"""

    def __init__(self, weights: str, device: str = "0",
                 imgsz: int = 416, conf: float = 0.3):
        from ultralytics import YOLO
        log.info(f"加载单字检测模型: {weights}")
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def detect(self, crop_img: np.ndarray) -> Tuple[str, float, int]:
        """
        检测裁剪图中的所有数字，按 x 坐标从左到右排序拼接。

        Returns:
            (reading, avg_conf, digit_count)
            reading: 拼接后的读数字符串，如 "334.42" 或 ""
            avg_conf: 平均置信度
            digit_count: 检测到的数字个数
        """
        results = self.model(crop_img, imgsz=self.imgsz, conf=self.conf,
                             device=self.device, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return "", 0.0, 0

        boxes = results[0].boxes
        detections = []
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            xc, yc, w, h = boxes.xywhn[i].cpu().numpy()
            conf = float(boxes.conf[i])
            detections.append((cls, xc, yc, w, h, conf))

        # 按 x 中心从左到右排序
        detections.sort(key=lambda d: d[1])

        chars = [DIGIT_NAMES[d[0]] for d in detections]
        reading = "".join(chars)
        avg_conf = sum(d[5] for d in detections) / len(detections) if detections else 0.0

        return reading, avg_conf, len(detections)



# 4. 时序投票机制（后端过滤）

class TemporalVoter:
    """
    滑动窗口时序投票器。
    逐位字符投票 + 置信度加权，剔除野值，结果稳定后触发输出。
    """

    def __init__(self, window_size: int = 10, stability_threshold: int = 5):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.queue: deque = deque(maxlen=window_size)
        self._last_stable: Optional[str] = None
        self._stable_count: int = 0

    def add(self, reading: Optional[str], confidence: float = 0.0):
        if reading and len(reading) > 0:
            self.queue.append((reading, confidence))

    def vote(self) -> Optional[str]:
        if len(self.queue) == 0:
            return None
        max_len = max(len(t) for t, _ in self.queue)
        result_chars = []
        for pos in range(max_len):
            char_votes: Dict[str, float] = {}
            for text, conf in self.queue:
                if pos < len(text):
                    ch = text[pos]
                    char_votes[ch] = char_votes.get(ch, 0.0) + conf
            if char_votes:
                result_chars.append(max(char_votes, key=char_votes.get))
        result = "".join(result_chars)
        if result == self._last_stable:
            self._stable_count += 1
        else:
            self._last_stable = result
            self._stable_count = 1
        return result

    def is_stable(self) -> bool:
        return self._stable_count >= self.stability_threshold

    def reset(self):
        self.queue.clear()
        self._last_stable = None
        self._stable_count = 0


# 5. 结果上报模块

class ResultReporter:
    """将最终读数保存为 JSON/CSV，预留无人机坐标接口。"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.json_path = os.path.join(output_dir, "readings.json")
        self.csv_path = os.path.join(output_dir, "readings.csv")
        self.records: List[dict] = []
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "reading", "confidence", "uav_lat", "uav_lon", "uav_alt"])

    def report(self, reading: str, confidence: float = 0.0,
               uav_lat: float = 0.0, uav_lon: float = 0.0, uav_alt: float = 0.0):
        record = {
            "timestamp": datetime.now().isoformat(),
            "reading": reading,
            "confidence": round(confidence, 4),
            "uav_position": {"latitude": uav_lat, "longitude": uav_lon, "altitude": uav_alt},
        }
        self.records.append(record)
        with open(self.json_path, "w") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [record["timestamp"], reading, confidence, uav_lat, uav_lon, uav_alt])
        log.info(f"读数上报: {reading} (置信度={confidence:.2f})")



# 6. 主 Pipeline 

class DigitalMeterPipeline:
    """
    数字仪表读数完整 Pipeline（双 YOLO）

    流程: 模糊过滤 → OBB 屏幕定位 → 裁剪 → 单字检测 → 排序拼接 → 时序投票

    Usage:
        pipeline = DigitalMeterPipeline(obb_weights="...", digit_weights="...")
        reading = pipeline.process_frame(frame)  # -> "33442" 或 None

        # 可选开启文件记录:
        pipeline = DigitalMeterPipeline(..., output_dir="results")
    """

    def __init__(self,
                 obb_weights: str = "runs/obb/train/weights/best.pt",
                 digit_weights: str = "runs/digit/train/weights/best.pt",
                 blur_threshold: float = 100.0,
                 obb_conf: float = 0.5,
                 digit_conf: float = 0.3,
                 obb_imgsz: int = 640,
                 digit_imgsz: int = 416,
                 voting_window: int = 10,
                 stability_threshold: int = 5,
                 output_dir: Optional[str] = None,
                 device: str = "0"):
        """
        Args:
            obb_weights: 阶段一 OBB 屏幕定位模型权重
            digit_weights: 阶段二单字检测模型权重
            blur_threshold: 拉普拉斯方差阈值，低于此值判定为模糊帧
            obb_conf / digit_conf: 两阶段检测置信度阈值
            obb_imgsz / digit_imgsz: 两阶段输入尺寸
            voting_window: 时序投票滑动窗口大小
            stability_threshold: 连续多少帧一致判定为稳定
            output_dir: 结果保存目录，None 则不保存文件（默认）
            device: 推理设备 ("0", "cpu")
        """
        self.blur_threshold = blur_threshold

        self.screen_detector = ScreenDetector(
            weights=obb_weights, device=device, imgsz=obb_imgsz, conf=obb_conf)
        self.digit_detector = DigitDetector(
            weights=digit_weights, device=device, imgsz=digit_imgsz, conf=digit_conf)
        self.voter = TemporalVoter(
            window_size=voting_window, stability_threshold=stability_threshold)
        self.reporter = ResultReporter(output_dir=output_dir) if output_dir else None

        self.frame_count = 0
        self.skip_blurry = 0
        self.skip_no_screen = 0

    def process_frame(self, frame: np.ndarray,
                      uav_lat: float = 0.0,
                      uav_lon: float = 0.0,
                      uav_alt: float = 0.0) -> Optional[str]:
        """
        处理单帧图像，返回识别到的读数字符串。

        Returns:
            读数字符串 (如 "33442")，识别失败返回 None。
            当 output_dir 已配置且读数稳定时，自动写入 JSON/CSV。
        """
        self.frame_count += 1

        # Step 1: 模糊过滤
        if is_frame_blurry(frame, self.blur_threshold):
            self.skip_blurry += 1
            return None

        # Step 2: OBB 屏幕定位 + 裁剪
        corners, crop = self.screen_detector.detect(frame)
        if corners is None or crop is None:
            self.skip_no_screen += 1
            return None

        # Step 3: 单字检测 + 排序拼接
        reading, avg_conf, digit_count = self.digit_detector.detect(crop)
        if not reading:
            return None

        log.debug(f"帧 {self.frame_count}: {reading} ({digit_count}位, conf={avg_conf:.2f})")

        # Step 4: 时序投票
        self.voter.add(reading, avg_conf)
        voted = self.voter.vote()

        # Step 5: 稳定后可选上报
        if voted and self.voter.is_stable() and self.reporter:
            self.reporter.report(voted, avg_conf, uav_lat, uav_lon, uav_alt)

        return voted


# ═══════════════════════════════════════════════════════
# 保留的兼容函数（供 debug 脚本等外部调用）
# ═══════════════════════════════════════════════════════

def get_corrected_screen(frame, obb_box, output_size=(300, 100)):
    """透视变换校正（兼容旧接口）。"""
    h, w = frame.shape[:2]
    pts = np.array(obb_box, dtype=np.float32).reshape(4, 2)
    if pts.max() <= 1.5:
        pts[:, 0] *= w
        pts[:, 1] *= h
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    dst_w, dst_h = output_size
    dst_pts = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    return cv2.warpPerspective(frame, M, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class YOLOBBDetector(ScreenDetector):
    """兼容旧名称。"""
    pass


__all__ = [
    "is_frame_blurry",
    "get_corrected_screen",
    "ScreenDetector",
    "DigitDetector",
    "TemporalVoter",
    "ResultReporter",
    "YOLOBBDetector",
    "DigitalMeterPipeline",
]
