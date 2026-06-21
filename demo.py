#使用示例
import cv2
from inference_pipeline import DigitalMeterPipeline

def main():
    # 初始化
    pipeline = DigitalMeterPipeline(
        obb_weights="runs/obb/train_3/weights/best.pt",
        digit_weights="runs/digit/train_3/weights/best.pt",
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