#!/usr/bin/env python3
"""
单字数字检测模型训练脚本
用于检测裁剪后数显长条图中的独立数字字符（0-9）。

模型: YOLO26n (Nano，轻量高速)
数据: dataset_digits/ (由 prepare_digit_dataset.py 生成)
特点: 长条小图，密集小目标，10 类纯数字
"""

import argparse
from ultralytics import YOLO


def train(args):
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,

        # ── 数据增强（针对小尺寸密集数字字符）──
        # 数字方向固定且 6/9 易混淆，所以禁用翻转和大角度旋转
        degrees=5.0,             # 轻微旋转，数字朝向基本固定
        perspective=0.0,         # 无透视，已由第一阶段校正
        scale=0.3,               # 适度缩放 (0.7~1.3)
        fliplr=0.0,              # 禁止水平翻转（数字序列有顺序）
        flipud=0.0,              # 禁止垂直翻转（6/9 会混淆）
        mosaic=0.5,              # 适度 Mosaic
        mixup=0.0,               # 关闭 mixup（小目标容易混叠）
        hsv_h=0.01,              # 极小色调变化
        hsv_s=0.3,               # 适度饱和度变化
        hsv_v=0.3,               # 适度明度变化
        translate=0.05,          # 极小平移，防止数字出界
        shear=0.0,               # 关闭剪切

        # ── 优化器 ──
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,

        # ── 其他 ──
        patience=25,
        save=True,
        val=True,
        exist_ok=args.exist_ok,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO 单字数字检测训练 (0-9)")
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--data", type=str,
                        default="dataset_digits/data.yaml",
                        help="单字数据集配置文件")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="输入尺寸 (裁剪图约 50-140px 宽, 416 足够)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="批大小 (-1 自动)")
    parser.add_argument("--device", type=str, default="0",
                        help="训练设备")
    parser.add_argument("--workers", type=int, default=8,
                        help="数据加载线程数")
    parser.add_argument("--project", type=str,
                        default="runs/digit",
                        help="保存目录")
    parser.add_argument("--name", type=str, default="train",
                        help="实验名称")
    parser.add_argument("--exist_ok", action="store_true",
                        help="允许覆盖已存在的实验目录（默认每次训练保存到新目录）")
    args = parser.parse_args()

    print(f"开始训练单字检测模型: model={args.model}, data={args.data}, imgsz={args.imgsz}")
    print(f"exist_ok={args.exist_ok} (若为 False，同名实验目录会自动追加数字后缀)")
    train(args)


if __name__ == "__main__":
    main()
