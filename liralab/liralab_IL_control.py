from liralab.model import *
import numpy as np
from spatialmath import SE3
import cv2
import torch
import torchvision.transforms as transforms
import roboticstoolbox as rtb
import numpy as np
import socket
import struct
import tensorflow as tf
from tensorflow import keras
from liralab.liralab_socket import LiralabSocket
from scipy.spatial.transform import Rotation as R
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# --------- NEURAL NETWORK
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# policy = ACTPolicy()
# policy.cuda()
# policy.load_state_dict(torch.load("experiments/policy_last.ckpt"))
# device = "cuda" if torch.cuda.is_available() else "cpu"
# policy.eval()

# ---------- NORMALIZATION
IMG_H = 256 # 480
IMG_W = 256 # 640
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],    std=[0.229, 0.224, 0.225])
qpos_mean = np.array([5.8461685, 0.49221334, 3.445083, 4.3707814, 1.298483, 4.7341285, 2.8364038 ], dtype=np.float32)
qpos_std = np.array([0.69456846, 0.19835953, 0.3600296,  0.43290123, 0.15821493, 0.42179197, 1.9556911 ], dtype=np.float32)
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

# ------------- MAIN
liralabSocket = LiralabSocket(5000)
cap = cv2.VideoCapture(2)
_, frame = cap.read()
plt.imshow(frame)
plt.show()
model = tf.keras.models.load_model(
    "./segmentation_models/unet_dnet121_case_v1.h5",
    # "./old_unet_dnet121/models/old_unet_dnet121_case_best_v1.h5",
    # custom_objects={'combined_loss': combined_loss, 'dice_coef': dice_coef, 'iou_score': iou_score},
    compile=False
)

# Register init pose
state = liralabSocket.read().split(';')[:-1]
T_belly_0 = get_tran_from_state(state)
T_0_belly = np.linalg.inv(T_belly_0)
liralabSocket.write("RUN")


while(True):
    # Read from socket
    state = liralabSocket.read().split(';')
    T_curr_0 = get_tran_from_state(state)
    T_curr_belly = np.dot(T_0_belly, T_curr_0)
    
    rpy = R.from_matrix(T_curr_belly[:3,:3]).as_euler('xyz').astype(np.float32)
    ee_pose_belly = np.concatenate([T_curr_belly[:3,3], rpy])

    # Capture frame for segmentation
    ret, frame = cap.read()
    if not ret: break 
    frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    frame = np.expand_dims(frame, axis=0)
    #
    mask = get_segmentation_mask(model,frame).squeeze()
    frame = frame.squeeze()[:,:,0].squeeze()
    input_image = append_frame_and_mask(frame, mask)

    # ------------------------------------------------------------------
    # TODO: Normalize ee_pose_belly with training dataset stats !!!!!! -
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # TODO: Denormalize ee_pose_belly with training dataset stats !!!!!! -
    # ------------------------------------------------------------------

# Capture camera frame
# inference: Segment frame and append channel
# Preprocess frame
# inference: ACT with [frame,x,y,z,roll,pitch,yaw]
# new_action -> transform from belly to world -> x,y,z,roll,pitch,yaw
# write socket