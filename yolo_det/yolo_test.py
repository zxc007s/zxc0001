from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("./model/yolov8l.pt")

    metrics = model.val(
        data="./yolo_cfg/mydata.yaml",   # 数据集配置文件
        project="./runs/val",       # 结果保存路径
        name="hat_det",             # 结果文件夹名称
        imgsz=500,                  # 输入图像尺寸
        batch=16,                   # 批处理大小
        conf=0.25,                  # 置信度阈值
        iou=0.6,                    # IoU阈值
        workers=8,                  # 数据加载的工作线程数
        device="0",                 # 使用的设备，"0"表示使用第一块GPU
        plots=True                  # 是否生成评估图表
    )

    print("map50-95:", metrics.box.map)  # map50-95
    print("map50:", metrics.box.map50)    # map50