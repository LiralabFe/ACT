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
from liralab_socket import LiralabSocket
from scipy.spatial.transform import Rotation as R

# --------- NEURAL NETWORK
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
policy = ACTPolicy()
policy.cuda()
policy.load_state_dict(torch.load("experiments/policy_last.ckpt"))
device = "cuda" if torch.cuda.is_available() else "cpu"
policy.eval()

# ---------- NORMALIZATION
IMG_H = 480
IMG_W = 640
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

# ------------- MAIN
liralabSocket = LiralabSocket(5000)

# Register init pose
state = liralabSocket.read().split(';')[:-1]
T_belly_0 = get_tran_from_state(state)
T_0_belly = np.linalg.inv(T_belly_0)
liralabSocket.write("RUN")


while(True):
    # Read from socket
    state = liralabSocket.read().split(';')[:-1]
    T_curr_0 = get_tran_from_state(state)
    T_curr_belly = np.dot(T_0_belly, T_curr_0)
    
    rpy = R.from_matrix(T_curr_belly[:3,:3]).as_euler('xyz').astype(np.float32)
    ee_pose_belly = np.concatenate([T_curr_belly[:3,3], rpy])

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