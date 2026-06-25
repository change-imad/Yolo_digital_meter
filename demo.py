#使用示例
import cv2
from inference_pipeline import DigitalMeterPipeline

def main():
    # 初始化
    pipeline = DigitalMeterPipeline(
        obb_weights="runs/obb/train_3/weights/best.pt",
        digit_weights="runs/digit/train_3/weights/best.pt",
    )

    pipeline = DigitalMeterPipeline(
    obb_weights="runs/obb/train_3/weights/best.engine",             # OBB 模型权重
    digit_weights="runs/digit/train_3/weights/best.engine",         # 单字检测权重
    blur_threshold=20.0,        # 拉普拉斯模糊阈值
    obb_conf=0.6,               # OBB 置信度阈值
    digit_conf=0.6,             # 单字检测置信度阈值
    obb_imgsz=640,              # OBB 输入尺寸
    digit_imgsz=416,            # 单字检测输入尺寸
    voting_window=8,           # 时序投票窗口
    stability_threshold=4,      # 稳定性判定帧数
    enhance_enabled=True,       # CLAHE 裁剪图增强预处理（默认开启）
    output_dir=None,            # None=不保存文件, "results"=写JSON/CSV
    device="0",                 # 推理设备
    )

    #调用本地摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("failed")
        return
    
    try:
        while True:
            ret,frame = cap.read()

            # 处理单帧，返回读数字符串
            reading = pipeline.process_frame(frame)
            print(reading)  

            cv2.putText(frame, f"Reading: {reading}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Digital Meter Detection', frame)

            #q键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("end")

if __name__ == "__main__":
    main()