"""import cv2
import numpy as np
import onnxruntime as ort
import time

# Configurazione
MODEL_PATH = "/home/legion/PycharmProjects/ACT/ACT_refactor/segmentation_models/jugular/SeResNet50_Phantom(Neck)_v4.onnx"
CAMERA_ID = "/home/legion/Downloads/jugular.mp4"
IMG_SIZE = 256

# ROI (x, y, width, height)
# None = utilizza l'intero frame
ROI = (380,100,350,350)

# Caricamento modello ONNX
session = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

input_name = session.get_inputs()[0].name

print("Modello caricato")
print("Provider:", session.get_providers())
print("Input:", session.get_inputs()[0].shape)
print("Output:", session.get_outputs()[0].shape)

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

    # Preprocessing IDENTICO al codice C#
    # Nessuna normalizzazione
    # Nessuna conversione BGR -> RGB
    input_tensor = img.astype(np.float32)

    # HWC -> CHW
    input_tensor = np.transpose(input_tensor, (2, 0, 1))

    # CHW -> NCHW
    input_tensor = np.expand_dims(input_tensor, axis=0)

    input_tensor = np.ascontiguousarray(input_tensor)

    # Inferenza
    start = time.perf_counter()

    output = session.run(None, {input_name: input_tensor})[0]

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
"""
import cv2

# Apri la webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Dimensioni frame originale
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ROI: (x, y, larghezza, altezza)
ROI = (335, 125, 1100 - 335, 600 - 125)

roi_width = ROI[2]
roi_height = ROI[3]

# Codec e file di output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "aorta_phantom_3.mp4",
    fourcc,
    30,
    (roi_width, roi_height)
)

if not out.isOpened():
    raise RuntimeError("Impossibile creare il file video.")

print("Registrazione avviata. Premi Q per terminare.")

try:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Estrai la ROI
        x, y, w, h = ROI
        frame = frame[y:y+h, x:x+w]

        # Registra il frame ritagliato
        out.write(frame)

        # Visualizza il frame
        cv2.imshow("Registrazione ROI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("Video salvato in aorta_phantom_1.mp4")