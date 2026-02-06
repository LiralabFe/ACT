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



# --------- SOCKET
HOST = "localhost"
PORT = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)

#print("In attesa di connessione...")
conn, addr = sock.accept()
print("Connesso da", addr)

# wrapper file-like (gestisce \n)
conn_file = conn.makefile("rwb")
cap = cv2.VideoCapture(2)
ret, frame = cap.read()

try:
    while True:
        data = conn.recv(28)  # 7 float32 = 28 byte
        if not data: break

        line = conn_file.readline()
        if not line: break

        qpos_str = line.decode().strip()
        qpos = [float(x) for x in qpos_str.split(";")]
        qpos = np.array([1,1,1,1,1,1,1], dtype=np.float32)
        qpos = (qpos - qpos_mean) / qpos_std
        qpos = torch.from_numpy(qpos).unsqueeze(0).to(device)  # [1,7]

        _, frame = cap.read()
        
        image_data = preprocess_frame(frame).to(device)
        output = policy(qpos,image_data)




finally:
    conn.close()
    sock.close()