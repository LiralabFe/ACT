from liralab.model import *
import numpy as np
from spatialmath import SE3
import cv2
import torchvision.transforms as transforms
# import roboticstoolbox as rtb
import numpy as np
import socket
import struct
import tensorflow as tf
import torch
from tensorflow import keras
import time
from liralab.liralab_socket import LiralabSocket
from scipy.spatial.transform import Rotation as R

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices([], 'GPU')
# --------- NEURAL NETWORK
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

policy = ACTPolicy()
policy.cuda()
policy.load_state_dict(torch.load("experiments/AAA/policy_last.ckpt"))
device = "cuda" if torch.cuda.is_available() else "cpu"
policy.eval()

# ---------- NORMALIZATION
IMG_H = 256 # 480
IMG_W = 256 # 640
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],    std=[0.229, 0.224, 0.225])
qpos_mean = np.array([0.00368241, -0.0334186, 0.02010795, 0.00242248, -0.01360511, -0.04260443], dtype=np.float32)
qpos_std = np.array([0.01278482, 0.01280284, 0.01, 0.03897353, 0.04673523, 0.04948749], dtype=np.float32)

def preprocess_frame(frame_bgr):
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # resize
    frame_rgb = cv2.resize(frame_rgb, (IMG_W, IMG_H))

    # to tensor
    frame = torch.from_numpy(frame_rgb).float()  # [H,W,C]
    frame = frame.permute(2, 0, 1)               # [C,H,W]

    frame /= 255.0
    frame = normalize(frame)

    return frame.unsqueeze(0)  # [1,C,H,W]

def get_tran_from_state(state):
    return np.array([
    [state[3],state[4],state[5],state[0]],
    [state[6],state[7],state[8],state[1]],
    [state[9],state[10],state[11],state[2]],
    [0,0,0,1]
    ], dtype=np.float32)

def get_segmentation_mask(model,frame):
    mask = model.predict(frame, verbose=1)[0]
    mask = (mask > 0.5).astype(np.uint8)
    return mask

def append_frame_and_mask(frame, mask):
    # Converti in numpy array
    r = np.array(frame, dtype=np.uint8)
    g = np.array(mask, dtype=np.uint8)
    b = np.zeros_like(r, dtype=np.uint8)

    # Stack nei canali RGB
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def transform_to_string(T):
    # Estrai traslazione
    x, y, z = T[0:3, 3]

    # Estrai rotazione 3x3 (flatten riga per riga)
    R = T[0:3, 0:3].flatten()

    values = [x, y, z] + R.tolist()

    return ";".join(f"{v:.4f}" for v in values)

# ------------- MAIN
liralabSocket = LiralabSocket(5000)
cap = cv2.VideoCapture(2)
ret, frame = cap.read()
while frame.max() == 0:
    ret, frame = cap.read()
    time.sleep(0.5)
plt.imshow(frame)
plt.show()
model = tf.keras.models.load_model("./segmentation_models/unet_dnet121_case_v1.h5",compile=False)

# Register init pose
state = liralabSocket.read().split(';')[:-1]
T_belly_0 = get_tran_from_state(state)
T_0_belly = np.linalg.inv(T_belly_0)
liralabSocket.write("RUN")


while(True):
    #------------------------#
    # Read state from socket #
    #------------------------#
    state = liralabSocket.read().split(';')
    T_curr_0 = get_tran_from_state(state)                                           # T from current position to origin
    T_curr_belly = np.dot(T_0_belly, T_curr_0)                                      # T from current position to belly
    rpy = R.from_matrix(T_curr_belly[:3,:3]).as_euler('xyz').astype(np.float32)     # roll pitch yaw
    ee_curr_belly = np.concatenate([T_curr_belly[:3,3], rpy])

    #--------------------------------#
    # Capture frame for segmentation #
    #--------------------------------#
    ret, frame = cap.read()                                                         # frame [w, h, 3]
    if not ret: break 
    frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)       # frame [255, 255, 3]
    frame = np.expand_dims(frame, axis=0)                                           # frame [1, 255, 255, 3]
    mask = get_segmentation_mask(model,frame).squeeze()                             # mask  [255, 255]
    frame = frame.squeeze()[:,:,0].squeeze()                                        # frame [255, 255]
    frame = append_frame_and_mask(frame, mask)                                      # frame [255, 255, 3] => [R: frame, G: mask, B: unused]

    #-------------------------#
    # Normalize input for ACT #
    #-------------------------#
    ACT_input_state = (ee_curr_belly - qpos_mean) / qpos_std
    ACT_input_state = torch.from_numpy(ACT_input_state).unsqueeze(0).to(device)
    ACT_input_image = preprocess_frame(frame).to(device)

    #---------------#
    # ACT Inference #
    #---------------#
    ACT_output_action = policy(ACT_input_state, ACT_input_image)

    #--------------------#
    # Denormalize output #
    #--------------------#
    ee_new_belly = ACT_output_action.cpu().squeeze()[0].detach().numpy() * qpos_std + qpos_mean
    # Bounding box rotation
    limit = 10 * np.pi / 180  # ≈ 0.174532925 rad
    ee_new_belly[3] = np.clip(ee_new_belly[3], -limit, limit)
    ee_new_belly[4] = np.clip(ee_new_belly[4], -limit, limit)
    ee_new_belly[5] = np.clip(ee_new_belly[5], -limit, limit)
    print(ee_new_belly[3:] * 180.0 /np.pi)
    eeR = R.from_euler('xyz', ee_new_belly[3:]).as_matrix()
    eeR = np.concatenate([eeR[0],eeR[1],eeR[2]])
    eeP = ee_new_belly[0:3]
    ee_new_belly = np.concatenate([eeP,eeR])
    ee_new_belly = get_tran_from_state(ee_new_belly)
    ee_new_0 = np.dot(T_belly_0 , ee_new_belly)

    #-----------------#
    # Write new state #
    #-----------------#

    liralabSocket.write(transform_to_string(ee_new_0))