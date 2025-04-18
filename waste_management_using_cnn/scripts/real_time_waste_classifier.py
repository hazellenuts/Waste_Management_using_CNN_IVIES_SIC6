import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

# ----------- 1. Load YOLOv5 dari PyTorch Hub ------------
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# ----------- 2. Load CNN kamu dari .pth ------------
# Buat ulang struktur model CNN (ResNet18)
cnn_model = resnet18(weights=ResNet18_Weights.DEFAULT)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)  # 2 kelas: Organik dan Recyclable

# Load state_dict ke model
cnn_model.load_state_dict(torch.load("../models/waste_classifier_cnn.pth", map_location=torch.device('cpu')))
cnn_model.eval()

# ----------- 3. Transformasi untuk input CNN ------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------- 4. Label kelas ------------
class_names = ['Organik', 'Anorganik']

# ----------- 5. Buka kamera ------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek dengan YOLOv5
    results = yolo_model(frame)
    boxes = results.xyxy[0]  # Format: [x1, y1, x2, y2, conf, class]

    for *box, conf, cls in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped = frame[y1:y2, x1:x2]

        try:
            # Ubah ke format untuk CNN
            image_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            input_tensor = transform(image_pil).unsqueeze(0)

            with torch.no_grad():
                output = cnn_model(input_tensor)
                pred = torch.argmax(output, 1).item()
                label = class_names[pred]

            # Gambar box dan label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)
        except:
            continue

    # Tampilkan hasil
    cv2.imshow("Waste Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
