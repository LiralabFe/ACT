import os
import numpy as np
import torch
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

def get_model(weights_path, device):
    """Inizializza il modello e carica i pesi addestrati."""
    print(f"Caricamento del modello dai pesi: {weights_path}")
    model = smp.UnetPlusPlus(
        encoder_name="densenet121",
        encoder_weights=None,  # Non serve scaricare ImageNet in inferenza, carichiamo i nostri
        in_channels=3,
        classes=2,             # Sfondo (0) e Giugolare (1)
        decoder_attention_type="scse"
    )
    
    # Carica i pesi
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Modalità inferenza
    
    return model

def get_transforms():
    """Restituisce le stesse identiche trasformazioni usate nel validation."""
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ])

# Load del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model("./segmentation_models/unetplusplus_imagenet_jugular.pth",device)

# Imposta la scheda di acquisizione (ad esempio, ID 0 per webcam o ID specifico per scheda di acquisizione)
# video_source = 0
# cap = cv2.VideoCapture(2)

# Crea una finestra per la trackbar
#cv2.namedWindow("Model Output View")
color = np.array([0, 0, 255], dtype='uint8')  # Rosso

# AP_1, SF_1, EM_1, MR_1, SF_2, SF_3, SF_4
EPISODE = "AAA_SF_4"
PATH = f"/home/legion/ROS/kinova_ws/AORTE/{EPISODE}/image/"
SAVE_PATH = f"/home/legion/ROS/kinova_ws/AORTE/{EPISODE}/mask/"

#if not os.path.isdir(SAVE_PATH):
#    os.mkdir(SAVE_PATH)

tot_images = len(os.listdir(PATH))
i = 0
cap = cv2.VideoCapture(0)
for image_path in os.listdir(PATH):
    print(f"Remaining {i}/{tot_images}")
    i += 1
    #ret, frame = cap.read()
    #if not ret:
    #    break  # Esce se il video è terminato
    frame = cv2.imread(PATH + image_path)

    # Crop dell'immagine
    # frame = frame[200:900, 475:1475]
    H, W, _ = frame.shape
    ori_frame = frame.copy()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_size = frame.shape[:2]  # (Altezza, Larghezza)
    
    # 3. Applica le trasformazioni
    transform = get_transforms()
    augmented = transform(image=frame)
    input_tensor = augmented['image'].unsqueeze(0).to(device)  # Aggiungi dimensione batch
    
    with torch.no_grad():
        output = model(input_tensor)
        # Prendi la classe con la probabilità più alta (0 o 1)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Ridimensionamento della maschera senza interpolazione
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Overlay con colore rosso per la maschera
    mask_color = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * color

    cv2.imwrite(SAVE_PATH + "mask_" + image_path, mask_color)

    print(f"Saved mask_{image_path} in {SAVE_PATH}")
    # Mostra il risultato finale
   
    # Premi "Invio" per uscire
    if cv2.waitKey(1) & 0xFF == 13:
        break

cap.release()
cv2.destroyAllWindows()
