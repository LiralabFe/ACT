from liralab.model import *
import numpy as np
from spatialmath import SE3
import cv2
import torch
import torchvision.transforms as transforms
import roboticstoolbox as rtb
import numpy as np
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

policy = ACTPolicy()
policy.cuda()
policy.load_state_dict(torch.load("experiments/AAA/policy_last.ckpt"))
policy.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"




# stessa normalize del dataset
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

IMG_H = 480
IMG_W = 640

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

dataset_stats = get_norm_stats("/home/legion/PycharmProjects/ACT/ACT_refactor/data/liralab/AAA/")
qpos = np.array([6.068143,0.302217,3.518075,4.509130,6.180255,5.217368,0.240442], dtype=np.float32)
qpos_mean = np.array(dataset_stats['qpos_mean'], dtype=np.float32)
qpos_std = np.array(dataset_stats['qpos_std'], dtype=np.float32)
qpos = (qpos - qpos_mean) / qpos_std
qpos = torch.from_numpy(qpos).unsqueeze(0).to(device)  # [1,7]

"""
cap = cv2.VideoCapture(1)

with torch.no_grad():
    while True:
        start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        
        image_data = preprocess_frame(frame).to(device)
        qpos_data = qpos.clone()  # [1,7]

        # action_data e is_pad → dummy
        # se il modello li richiede nel forward
        action_dummy = None
        is_pad_dummy = None
        
        output = policy(qpos_data,image_data)
        end = time.perf_counter()
        #print(end - start)
        # usa output (azioni predette ecc.)
        cv2.imshow("webcam", frame)
        #print(frame.size)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


quit()
"""
# ---------- parametri ----------

#urdf_path = "/home/legion/ROS/kinova_ws/src/ros2_kortex/kortex_description/robots/gen3.urdf"

robot = rtb.models.KinovaGen3()

#A = robot.fkine([6.055314,0.467064,3.701667,4.728049,5.991168,5.213708,0.318628]) #robot.ik_LM([6.084512,0.608651,3.264462,4.638094,1.374158,4.659789,1.450143]) # robot.ik_LM(Tep)         # solve IK
#B = robot.fkine([6.055314,0.467064,3.701667,4.728049,5.991168,5.213708,0.318628])#robot.ik_LM([6.0662956,0.5701123,3.3024302,4.7745996,1.4717325,4.7106547,3.3542852 ])

#print(A.t)
#print(B.t)
#print(np.linalg.norm((A.t - B.t)))

#qt = rtb.jtraj(robot.qr,[6.055314,0.467064,3.701667,4.728049,5.991168,5.213708,0.318628], 150)
#robot.plot(qt.q, backend='pyplot', movie='panda1.gif')


"""s
# ----------- ----------------------------------------------------- --------------
# ----------- PRINTA LA TRAIETTORIA PREDETTA A PARTIRE DA UNO STATO --------------
# ----------- ----------------------------------------------------- --------------

frame = cv2.imread("/home/legion/ROS/kinova_ws/AORTE/AAA_MR_1/image/img_0.png")
frame = preprocess_frame(np.array(frame)).to(device)
qpos = np.array([6.068143,0.302217,3.518075,4.509130,6.180255,5.217368,0.240442], dtype=np.float32)
qpos = (qpos - qpos_mean) / qpos_std
qpos = torch.from_numpy(qpos).unsqueeze(0).to(device) 

Q = policy(qpos,frame).squeeze()
Q = Q.cpu().detach().numpy() * qpos_std + qpos_mean

from scipy.signal import savgol_filter
Q = savgol_filter(Q, window_length=11, polyorder=3, axis=0)
robot.plot(Q, backend='pyplot')
"""
# ----------- ------------------------------------------------------------------------------------------------ --------------
# ----------- PRINTA LE TRAIETTORIE PRENDENDO COME INPUT LO STATO PREDETTO E L'IMMAGINE SUCCESSIVA DEL DATASET --------------
# ----------- ----------------------------------------------------- ------------------------------------------ --------------
qpos = np.array([6.068143,0.302217,3.518075,4.509130,6.180255,5.217368,0.240442], dtype=np.float32)
next_q = (qpos - qpos_mean) / qpos_std
next_q = torch.from_numpy(next_q).unsqueeze(0).to(device)
Q = np.empty((0,7))
Q = np.vstack((Q,next_q.cpu().detach().numpy() * qpos_std + qpos_mean))
i = 0
for img in os.listdir("/home/legion/ROS/kinova_ws/AORTE/AAA_MR_1/image/"):
    i += 1
    if i % 20 == 0: print(i)
    frame = cv2.imread("/home/legion/ROS/kinova_ws/AORTE/AAA_MR_1/image/" + img)
    frame = preprocess_frame(np.array(frame)).to(device)
    out = policy(next_q, frame)
    Q = np.vstack((Q,out.cpu().squeeze()[0].detach().numpy()  * qpos_std + qpos_mean))
    next_q = out.cpu().detach().squeeze()[0].to(device).unsqueeze(0)


from scipy.signal import savgol_filter
Q = savgol_filter(Q, window_length=11, polyorder=3, axis=0)
robot.plot(Q, backend='pyplot')


"""
# ----------- PRINTA LA TRAIETTORIA DAL FILE CSV --------------

import csv
csv_path = "/home/legion/ROS/kinova_ws/AORTE/AAA_JR_2/AAA_JR_2.csv"  # <-- cambia path

data = []

with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)  # salta header

    for row in reader:
        values = [float(x) for x in row[1:8]]

        data.append(values)

robot.plot(np.array(data, dtype=np.float32), backend='pyplot')
"""