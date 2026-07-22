import os
import glob
import time
import numpy as np
import torch
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from segmentation_models.hardsmeg.HarDNet_MSEG.lib.HarDMSEG import HarDMSEG


def get_model_unetplusplus(weights_path, device):
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

def get_model_hardsmeg(weights_path, device):
    model = HarDMSEG()
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

EPISODE = "TEST"
cartella = f"/home/legion/ROS/kinova_ws/{EPISODE}/image/"
cartella_mask = f"/home/legion/ROS/kinova_ws/{EPISODE}/mask/"
ultimo_file_mostrato = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Runnung on {device}")
# model = get_model_unetplusplus("./segmentation_models/unetplusplus_imagenet_jugular.pth",device)
model = get_model_hardsmeg("./segmentation_models/hardsmeg/hardnet68.pth", device)

color = np.array([0, 0, 255], dtype='uint8')  # Rosso
if not os.path.isdir(cartella_mask):
        os.mkdir(cartella_mask)

while True:
    # trova tutti i jpg/jpeg
    files = glob.glob(os.path.join(cartella, "*.png"))
    if files:
        # file più recente
        ultimo_file = max(files, key=os.path.getmtime)

        # se è diverso da quello già mostrato
        if ultimo_file != ultimo_file_mostrato:
            frame = cv2.imread(ultimo_file)
            print(ultimo_file)

            if frame is not None:
                ultimo_file_mostrato = ultimo_file
                H, W, _ = frame.shape
                ori_frame = frame.copy()

                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_size = frame.shape[:2]  # (Altezza, Larghezza)
                
                # 3. Applica le trasformazioni
                transform = get_transforms()
                augmented = transform(image=frame)
                input_tensor = augmented['image'].unsqueeze(0).to(device)  # Aggiungi dimensione batch
                
                with torch.no_grad():
                    output = model(input_tensor)

                    if isinstance(model, HarDMSEG):
                        if isinstance(output, tuple):
                            output = output[0]
                        prob = torch.sigmoid(output)
                        mask = (prob > 0.5).float().squeeze().cpu().numpy()
                        output = model(input_tensor)
                    # Prendi la classe con la probabilità più alta (0 o 1)
                    elif isinstance(model, smp.UnetPlusPlus):
                        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                    else:
                        print("Model is not a known instance")

                # Ridimensionamento della maschera senza interpolazione
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

                # Overlay con colore rosso per la maschera
                mask_color = (np.repeat(mask[:, :, np.newaxis], 3, axis=2) * color).astype("uint8") * 255.0
                cv2.imshow("Ultima immagine", (mask_color + ori_frame)/255.0)

    # necessario per aggiornare la finestra
    if cv2.waitKey(100) & 0xFF == 27:  # ESC per uscire
        break

    time.sleep(0.1)

cv2.destroyAllWindows()

"""
Format Video Capture:
	Width/Height      : 1920/1080
	Pixel Format      : 'MJPG' (Motion-JPEG)
	Field             : None
	Bytes per Line    : 0
	Size Image        : 4147200
	Colorspace        : sRGB
	Transfer Function : Rec. 709
	YCbCr/HSV Encoding: ITU-R 601
	Quantization      : Default (maps to Full Range)
	Flags             : 

    
    Streaming Parameters Video Capture:
	Capabilities     : timeperframe
	Frames per second: 30.000 (30/1)
	Read buffers     : 0


Format Video Capture:
	Width/Height      : 1280/720
	Pixel Format      : 'YUYV' (YUYV 4:2:2)
	Field             : None
	Bytes per Line    : 2560
	Size Image        : 1843200
	Colorspace        : sRGB
	Transfer Function : Rec. 709
	YCbCr/HSV Encoding: ITU-R 601
	Quantization      : Default (maps to Limited Range)
	Flags             : 

    	Capabilities     : timeperframe
	Frames per second: 10.000 (10/1)
	Read buffers     : 0

"""