import os
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

def sort_by_final_number(file_list):
    return sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Load del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = get_model_unetplusplus("./segmentation_models/unetplusplus_imagenet_jugular.pth",device)
model = get_model_hardsmeg("./segmentation_models/hardsmeg/hardnet68.pth", device)

# Imposta la scheda di acquisizione (ad esempio, ID 0 per webcam o ID specifico per scheda di acquisizione)
# video_source = 0
# cap = cv2.VideoCapture(0)

# Crea una finestra per la trackbar
#cv2.namedWindow("Model Output View")
color = np.array([0, 0, 255], dtype='uint8')  # Rosso

EPISODES_PATH = "/home/legion/ROS/kinova_ws/AORTE"
episodes_list = [f for f in os.listdir(EPISODES_PATH) if os.path.isdir(os.path.join(EPISODES_PATH, f))]
e = 0
for EPISODE in sort_by_final_number(episodes_list):
    print("#"*50)
    print("#"*10 + f" PROCESSING EPISODE {EPISODE} " + "#"*10)
    print("#"*50)
    print(f"###### DONE {e}/{len(episodes_list)} episodes")
    e += 1
    PATH = f"{EPISODES_PATH}/{EPISODE}/image/"
    SAVE_PATH = f"{EPISODES_PATH}/{EPISODE}/mask/"

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    tot_images = len(os.listdir(PATH))
    i = 0

    for image_path in sort_by_final_number(os.listdir(PATH)):
        if i % 100 == 0:
            print(f"DONE {i}/{tot_images} for episode {EPISODE}")
        i += 1
        #ret, frame = cap.read()
        #if not ret:
        #    break  # Esce se il video è terminato
        frame = cv2.imread(PATH + image_path)

        # Crop dell'immagine
        # frame = frame[200:900, 475:1475]
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

        cv2.imwrite(SAVE_PATH + "mask_" + image_path, mask_color)
        cv2.imshow(EPISODE,(mask_color + ori_frame)/255.0)
        # Mostra il risultato finale
    
        # Premi "Invio" per uscire
        if cv2.waitKey(1) & 0xFF == 13:
            break
    cv2.destroyAllWindows()

print("\n\nFINISH\n\n")