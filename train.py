from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Pretrained YOLO11 Nano

results = model.train(
    data="C:\\Users\\tamil selvan\\OneDrive\\Desktop\\Milk Packet Detection\\data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)