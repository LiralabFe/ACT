import os
import glob
import time
import numpy as np
import torch
import cv2
from liralab.utils.segmentator import Segmentator

EPISODE = "TEST"
cartella = f"/home/legion/ROS/kinova_ws/{EPISODE}/image/"
cartella_mask = f"/home/legion/ROS/kinova_ws/{EPISODE}/mask/"
ultimo_file_mostrato = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Runnung on {device}")

color = np.array([0, 0, 255], dtype='uint8')  # Rosso
if not os.path.isdir(cartella_mask):
        os.mkdir(cartella_mask)

seg = Segmentator("./segmentation_models/hardsmeg/hardnet68.pth", "HarDMSEG")

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
                mask = seg.get_segmented_mask(frame)

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