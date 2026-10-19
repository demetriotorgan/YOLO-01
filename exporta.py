from ultralytics import YOLO

# Baixa o modelo leve pré-treinado do YOLOv8 Nano
model = YOLO('yolov8n.pt')

# Exporta especificamente para o formato ONNX que o Node.js vai ler
model.export(format='onnx')