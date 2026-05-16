#!/usr/bin/env python3
"""
YOLO26-OBB 旋转目标检测训练脚本
用于训练数字仪表屏幕定位模型。

模型: yolo26n-obb.pt (Nano 版本，适合边缘部署)
数据增强: 针对无人机户外场景配置强增强策略
"""

import argparse
import sys, os
# 移除当前目录，避免本地 ultralytics/ 目录遮蔽 pip 安装的包
sys.path = [p for p in sys.path if p != '' and p != os.getcwd()]
from ultralytics import YOLO



def train(args):
    # 加载预训练模型
    model = YOLO(args.model)

    # 训练参数 —— 针对无人机巡检场景的强数据增强
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,

        # ── 强数据增强参数（对抗无人机户外光照变化和多视角）──
        degrees=45.0,           # 大角度旋转增强
        perspective=0.0015,     # 透视变换，模拟无人机不同俯仰角
        scale=0.9,              # 缩放范围 (0.1~1.9)
        mixup=0.3,              # MixUp 混合增强
        fliplr=0.5,             # 水平翻转
        flipud=0.1,             # 轻微垂直翻转
        mosaic=1.0,             # Mosaic 数据增强
        hsv_h=0.02,             # 色调变化
        hsv_s=0.7,              # 饱和度变化（户外光照变化大）
        hsv_v=0.4,              # 明度变化
        translate=0.2,          # 平移增强
        shear=10.0,             # 剪切增强

        # ── OBB 专用 ──
        task="obb",

        # ── 优化器 ──
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        cos_lr=True,

        # ── 其他 ──
        patience=25,
        save=True,
        val=True,
        exist_ok=True,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO26-OBB 数字仪表屏幕定位训练")
    parser.add_argument("--model", type=str,
                        default="yolo26n-obb.pt",
                        help="预训练模型路径")
    parser.add_argument("--data", type=str,
                        default="dataset/data.yaml",
                        help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=-1,
                        help="批大小 (-1 为自动)")
    parser.add_argument("--device", type=str, default="0",
                        help="训练设备 (0, cpu, 0,1)")
    parser.add_argument("--workers", type=int, default=8,
                        help="数据加载线程数")
    parser.add_argument("--project", type=str,
                        default="runs/obb",
                        help="保存目录")
    parser.add_argument("--name", type=str, default="train",
                        help="实验名称")
    args = parser.parse_args()

    print(f"开始训练: model={args.model}, data={args.data}")
    train(args)


if __name__ == "__main__":
    main()
