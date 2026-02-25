import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
os.environ["TF_USE_LEGACY_KERAS"] = "1"


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Combined Loss
def combined_loss(y_true, y_pred):
    return keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred) + dice_loss(y_true, y_pred)

# Load del modello
model = tf.keras.models.load_model(
    "./segmentation_models/unet_dnet121_case_v1_AORTA.h5",
    #"./old_unet_dnet121/models/old_unet_dnet121_case_best_v1.h5",
    # custom_objects={'combined_loss': combined_loss, 'dice_coef': dice_coef, 'iou_score': iou_score},
    compile=False
)
model.summary()

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

    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # if frame.ndim == 2:
    #     frame = np.expand_dims(frame, axis=-1)
    # if frame.ndim == 3 and frame.shape[-1] != 3:
    #     frame = np.repeat(frame, 3, axis=-1)
    # frame = tf.cast(frame, tf.float32) / 255.0

    frame = np.expand_dims(frame, axis=0)

    # Predizione della maschera
    mask = model.predict(frame, verbose=1)[0]
    mask = (mask > 0.5).astype(np.uint8)

    # Ridimensionamento della maschera senza interpolazione
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # **DEBUG: Mostra la maschera grezza**
    #cv2.imshow("Raw Mask Prediction", mask * 255)

    # Overlay con colore rosso per la maschera
    mask_color = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * color
    overlay = cv2.addWeighted(ori_frame, 0.8, mask_color, 0.2, 0)

    cv2.imwrite(SAVE_PATH + "mask_" + image_path, mask_color)

    print(f"Saved mask_{image_path} in {SAVE_PATH}")
    # Mostra il risultato finale
    cv2.imshow("Segmentation", mask_color)
    # Premi "Invio" per uscire
    if cv2.waitKey(1) & 0xFF == 13:
        break

cap.release()
cv2.destroyAllWindows()
