#!/usr/bin/env python3
"""
模型导出脚本：将训练好的 YOLO OBB .pt 权重转换为 TensorRT .engine 格式。
支持 x86_64 (PC) 和 aarch64 (Jetson) 平台。
"""

import platform
import argparse
import sys, os
# 移除当前目录，避免本地 ultralytics/ 目录遮蔽 pip 安装的包
sys.path = [p for p in sys.path if p != '' and p != os.getcwd()]
from ultralytics import YOLO


def export(args):
    model = YOLO(args.weights)

    arch = platform.machine()
    print(f"当前平台架构: {arch}")

    if arch == "aarch64":
        # Jetson 平台：导出 TensorRT FP16
        print("检测到 Jetson (aarch64)，导出 TensorRT FP16 engine...")
        model.export(
            format="engine",
            half=True,
            imgsz=args.imgsz,
            device=0,
        )
    else:
        # x86_64 PC 平台：优先 TensorRT，回退 ONNX
        try:
            print("检测到 PC (x86_64)，尝试导出 TensorRT FP16 engine...")
            model.export(
                format="engine",
                half=True,
                imgsz=args.imgsz,
                device=0,
            )
        except Exception as e:
            print(f"TensorRT 导出失败 ({e})，回退到 ONNX...")
            model.export(
                format="onnx",
                half=True,
                imgsz=args.imgsz,
                simplify=True,
            )
            print("已导出 ONNX 模型。")


def main():
    parser = argparse.ArgumentParser(description="YOLO OBB 模型导出")
    parser.add_argument("--weights", type=str, required=True,
                        help="训练好的 .pt 权重文件路径")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸")
    args = parser.parse_args()

    export(args)


if __name__ == "__main__":
    main()
