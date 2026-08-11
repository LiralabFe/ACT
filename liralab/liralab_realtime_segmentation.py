import cv2
import numpy as np
import torch
from liralab.utils.segmentator import Segmentator

# Device PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modello
seg = Segmentator(
    "./segmentation_models/hardsmeg/hardnet68.pth",
    "HarDMSEG"
)

color = np.array([0, 0, 255], dtype=np.uint8)

cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Impossibile aprire il device 1")

# ---------------- ROI ----------------
ROI_X = 150
ROI_Y = 80
ROI_W = 320
ROI_H = 320
# -------------------------------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # Assicura che la ROI sia valida
    x = max(0, ROI_X)
    y = max(0, ROI_Y)
    w = min(ROI_W, W - x)
    h = min(ROI_H, H - y)

    # Estrai ROI
    roi = frame[y:y+h, x:x+w]

    # Segmentazione della sola ROI
    mask = seg.get_segmented_mask(roi)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Overlay rosso
    mask_color = (
        np.repeat(mask[:, :, None], 3, axis=2) * color
    ).astype(np.uint8)

    # Copia dell'immagine originale
    output = frame.copy()

    # Fusione solo nella ROI
    output[y:y+h, x:x+w] = cv2.addWeighted(
        roi,
        0.7,
        mask_color,
        0.3,
        0
    )

    # Disegna il rettangolo della ROI
    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Segmentation", output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()