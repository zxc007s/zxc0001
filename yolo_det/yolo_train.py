from ultralytics import YOLO

model= YOLO("./yolo_cfg/yolov8l.yaml")

if __name__ == "__main__":

    results = model.train(
        data="./yolo_cfg/mydata.yaml",
        epochs=3,
        imgsz=500,
        batch=12,
        device=0,
        project="./results",
        name="had_det",
        save_period=1,
        workers=3,
        lr0=0.01,
        pretrained=False,
        amp=True
)


