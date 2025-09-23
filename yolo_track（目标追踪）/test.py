import cv2
import numpy as np
from ultralytics import YOLO


classid_to_name = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}


class VideoROICounter:
    #用来处理视频中的ROI区域内的目标检测和计数
    def __init__(self, model_path="./model/yolov8l.pt", mask_path=None, classes=None):
        self.model = YOLO(model_path)       # 加载模型
        self.counters = {}                   # 轨迹计数器
        self.track_history = {}             # 记录每个轨迹的历史位置
        self.counted_tracks = set()             # 记录已经计数的轨迹id
        self.track_len = 20                 # 轨迹长度限制
        self.track_max = 30                 # 最大轨迹数
        self.classes = classes               # 只检测指定类别
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 加载ROI 遮罩
        for class_id in classes:            # 初始化计数器
            self.counters[class_id] = 0         # 初始化每个类别的计数器为0

    # 绘制ROI遮罩在视频帧上
    def draw_roi_mask(self, frame, roi_alpha = 0.3):   #用于绘制ROI遮罩，roi_alpha 控制遮罩透明度
        # 调整mask大小以匹配当前帧
        height, width = frame.shape[:2]
        mask_resized = cv2.resize(self.mask, (width, height))

        # 创建彩色ROI 覆盖层, 创建一个绿色的区域
        roi_overlay = np.zeros_like(frame)        # 创建一个与帧大小相同的黑色图像
        roi_overlay[mask_resized > 127] = [0, 255, 0]  # 将ROI区域设置为绿色

        # 将彩色ROI与原图进行比例叠加
        frame = cv2.addWeighted(frame, 1 - roi_alpha, roi_overlay, roi_alpha, 0)

        # 绘制ROI轮廓
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        return frame

    # 处理单帧图像
    def process_frame(self, frame):
        # 利用mask进行图像处理
        frame_mask = cv2.bitwise_and(frame, frame, mask=self.mask)       # 应用遮罩，只保留ROI区域
        frame = self.draw_roi_mask(frame)                                # 绘制ROI遮罩，便于可视化

        # 进行检测和追踪
        results = self.model.track(frame_mask, classes=self.classes, persist=True)   #persist=True 保持追踪器状态,若为False则每帧重新初始化
        # results = self.model.track(frame, classes=self.classes, persist=True)
        # 显示计数结果（大字体显示）
        for i, (class_id, count) in enumerate(self.counters.items()):
            class_name = classid_to_name[class_id]
            cv2.putText(frame, f"{class_name}: {count}", (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # 检查是否有检测结果
        if results[0].boxes is None or results[0].boxes.id is None:
            return frame
        # 获取检测结果
        boxes = results[0].boxes.xywh.cpu()      # 获取检测框 (x, y, w, h)
        track_ids = results[0].boxes.id.cpu().tolist()  # 获取追踪ID，每个检测框对应一个ID
        class_ids = results[0].boxes.cls.cpu().tolist()  # 获取类别ID，每个检测框对应一个类别

        # 处理每个检测到的对象
        for box, track_id, class_id in zip(boxes, track_ids, class_ids): # 遍历每个检测框的坐标，id和类别
            # 获取信息
            x, y, w, h = box                # 获取检测框的坐标
            center = (int(x), int(y))       # 计算检测框的中心点
            track_id = int(track_id)   # 确保track_id是整数
            # 更新轨迹历史记录器
            if track_id not in self.track_history:   # 检查是否已有该track_id的历史记录
                self.track_history[track_id] = []    # 如果没有，则初始化一个空列表
                print(f"新轨迹ID: {track_id}, 类别: {classid_to_name[class_id]}")  # 打印新轨迹ID和类别
            track = self.track_history[track_id]        # 获取该track_id的历史记录
            track.append(center)                    # 将当前中心点添加到该track_id的历史记录中
            self.track_history[track_id] = track  # 更新该track_id的历史记录
            # 限制轨迹长度，以及计数逻辑
            if len(track) > self.track_len:              # 限制历史记录的长度，避免过长
                track.pop(0)
            if track_id not in self.counted_tracks:      # 如果该track_id没有出现，则加1
                self.counters[class_id] += 1
            self.counted_tracks.add(track_id)           # 将id记录在已见过的id集合中

            # 绘制轨迹
            # if len(track) > 1:              # 只有当轨迹点数大于1时才绘制轨迹
            #     points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))    # 将轨迹点转换为适合绘制的格式
            #     color = (0, 255, 255)       # 轨迹颜色为黄色
            #     cv2.polylines(frame, [points], False, color, 2) # 绘制轨迹线

            # 绘制检测框
            box_color = (0, 255, 0)         # 默认框颜色为绿色
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)),
                          (int(x + w / 2), int(y + h / 2)), box_color, 2)   # 绘制检测框

            # 显示track_id和类别
            cv2.putText(frame, str(track_id)+str(classid_to_name[class_id]),(int(x - w / 2), int(y - h / 2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # 限制轨迹数量，如果超过最大数量，则删除最早的轨迹
        if len(self.counted_tracks) > self.track_max:
            track_id_min = min(self.counted_tracks)  # 找到最小的track_id
            del self.track_history[track_id_min]     # 删除最小的track_id的
            self.counted_tracks.remove(track_id_min)  # 从已见过的id集合中删除
        return frame

    # 处理视频文件
    def process_videos(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)          # 打开视频文件
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # 获取视频高度
        fps = cap.get(cv2.CAP_PROP_FPS)                     # 获取视频帧率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')            # 视频编码格式
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))    # 创建视频写入对象

        # 处理视频
        while cap.isOpened():                       # 检查视频是否打开
            ret, frame = cap.read()             # 读取一帧
            if not ret:                         # 如果没有读取到帧，则退出循环
                break
            processed_frame = self.process_frame(frame)     # 处理当前帧
            cv2.imshow("Video ROI Counter", processed_frame)    # 显示处理后的帧
            out.write(processed_frame)      # 写入处理后的帧到输出视频
            if cv2.waitKey(1) & 0xFF == ord('q'):   # 如果按下 'q' 键，则退出循环
                break

        # 释放资源
        cap.release()       # 释放视频捕获对象
        out.release()       # 释放视频写入对象
        cv2.destroyAllWindows()     # 关闭所有OpenCV窗口

        # 最终统计
        print(f"\n=== 最终统计 ===")
        for class_id, count in self.counters.items():
            print(f"{classid_to_name[class_id]}: {count}")


if __name__ == "__main__":
    mask_path = "./data/img.jpg"
    video_path = "./data/test2.mp4"
    output_path = "./output/test2.mp4"

    roi_counter = VideoROICounter(
        model_path="./model/yolov8l.pt",
        mask_path=mask_path,
        classes=[49]
    )
    roi_counter.process_videos(video_path, output_path)