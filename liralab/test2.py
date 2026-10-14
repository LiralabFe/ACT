import cv2
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import FlexibleUNet
import time

# Configurazione
MODEL_PATH = "/home/legion/PycharmProjects/ACT/ACT_refactor/segmentation_models/jugular/SeResNet50_Phantom(Neck)_v4.pth"
CAMERA_ID = "/home/legion/Downloads/jugular.mp4"
IMG_SIZE = 256

# ROI (x, y, width, height)
# None = utilizza l'intero frame
ROI = (380,100,350,350)

# Architettura verificata sui pesi .pth e confrontata con il modello ONNX.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlexibleUNet(in_channels=3,out_channels=2,
    backbone="efficientnet-b2",
    pretrained=False,
    is_pad=False,
    pre_conv=None,
)
model.segmentation_head = nn.Conv2d(16, 3, kernel_size=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.to(device).eval()

# Queste trasformazioni sono incorporate nell'ONNX, ma non nei pesi .pth.
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

print("Modello caricato")
print("Dispositivo:", device)
print("Input:", [1, 3, IMG_SIZE, IMG_SIZE])
print("Output:", [1, 3, IMG_SIZE, IMG_SIZE])

# Apertura camera
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Impossibile aprire la camera")

print("Inferenza avviata. Premi Q per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Torna al primo frame
        continue

    # ROI
    if ROI is not None:
        x, y, w, h = ROI
        frame = frame[y:y+h, x:x+w]

    # Resize a 256x256
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # Stesso input esterno dell'ONNX: BGR float32, senza conversione RGB.
    input_tensor = img.astype(np.float32)

    # HWC -> CHW
    input_tensor = np.transpose(input_tensor, (2, 0, 1))

    # CHW -> NCHW
    input_tensor = np.expand_dims(input_tensor, axis=0)

    input_tensor = np.ascontiguousarray(input_tensor)
    input_tensor = torch.from_numpy(input_tensor).to(device)

    # Riproduci la normalizzazione interna del modello ONNX.
    input_tensor = (input_tensor / 255.0 - mean) / std

    # Inferenza
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.inference_mode():
        output = torch.softmax(model(input_tensor), dim=1).cpu().numpy()

    inference_ms = (time.perf_counter() - start) * 1000

    # Output: [1, 3, 256, 256]
    # Canali: background, giugulare, carotide
    mask = np.argmax(output, axis=1)[0].astype(np.uint8)

    # Maschera colorata
    mask_color = np.zeros_like(img)

    mask_color[mask == 1] = (255, 0, 0)  # Giugulare: blu
    mask_color[mask == 2] = (0, 0, 255)  # Carotide: rosso

    # Overlay
    overlay = cv2.addWeighted(img, 0.6, mask_color, 0.4, 0)

    # Mostra tempo inferenza
    cv2.putText(
        overlay,
        f"Inference: {inference_ms:.1f} ms",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Visualizzazione
    cv2.imshow("Camera", img)
    cv2.imshow("Segmentation", overlay)
    cv2.imshow("Mask", mask_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
