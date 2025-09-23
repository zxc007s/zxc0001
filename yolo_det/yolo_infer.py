from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("./model/yolov8l.pt")
#
# # Display model information (optional)
# model.info()
#
# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model.predict("./data/car.jpg",conf=0.25,save=True,device=0,iou=0.5)
print(results)