import numpy as np
import cv2
import torchvision.transforms as transforms
# import roboticstoolbox as rtb
import numpy as np
import socket
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import time
from liralab.liralab_socket import LiralabSocket
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path
from collections import deque
from liralab.utils.segmentator import Segmentator
import matplotlib.pyplot as plt

class liralabGRIDControl:
    def __init__(self, APP : str):
        assert APP in ['AORTA', 'JUGUL', 'CAROT'], f"{APP} not in ['AORTA', 'JUGUL', 'CAROT']"

        self.app = APP
        self.models = {
            'AORTA' : {
                'SEG' : "/home/legion/PycharmProjects/ACT/ACT_refactor/segmentation_models/hardsmeg/hardnet68.pth",
                'SEG_MODEL' : "HarDMSEG",
                'MIN_SUCCESS_FRAMES' : 15,
                'BUFFER_FRAMES' : 100,
                'FRAME_TO_SUCCESS' : 15,
                'MIN_DIAMETER' : 9,
                'PIXEL_TO_MM' : 1.0/1.8, # 1.8 pixels = 1mm nella ROI attuale ( Zoom: 27 Hz)
            },
            'JUGUL' : {
                'ACT' : "experiments/JVP/policy_last.ckpt",
                'SEG' : "segmentation_models/unetplusplus_imagenet_jugular.pth",
            },
            'CAROT' : {
                'ACT' : "experiments/CAS/policy_last.ckpt",
                'SEG' : "segmentation_models/unetplusplus_imagenet_jugular.pth",
            },
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---------- SEGMENTATOR
        self.segmentator = None

        # ---------- NORMALIZATION
        self.IMG_H = 256 # 480
        self.IMG_W = 256 # 640
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std)
        self.seg_normalization = A.Compose([A.Normalize(mean, std),A.ToFloat(max_value=255.0),ToTensorV2()])

        # ---------- INIT
        self.liralabSocket = LiralabSocket(5024)
        self.cap = cv2.VideoCapture(4)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = self.cap.read()
        while frame is None or frame.max() == 0:
            ret, frame = self.cap.read()
            time.sleep(0.5)
        frame = cv2.resize(frame, (640, 360))
        print(frame.shape)
        plt.imshow(frame)
        plt.show()

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.zeros((256,256,3)))

    def preprocess_frame(self,frame_bgr):
        # BGR -> RGB
        frame_rgb = frame_bgr#cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # resize
        frame_rgb = cv2.resize(frame_rgb, (self.IMG_W, self.IMG_H))

        # to tensor
        frame = torch.from_numpy(frame_rgb).float()  # [H,W,C]
        frame = frame.permute(2, 0, 1)               # [C,H,W]

        frame /= 255.0
        frame = self.normalize(frame)
        return frame.unsqueeze(0)  # [1,C,H,W]

    def append_frame_and_mask(self, frame, mask):
        # Converti in numpy array
        r = np.array(frame, dtype=np.uint8)
        g = np.array(mask, dtype=np.uint8)
        b = np.array(mask, dtype=np.uint8) # np.zeros_like(r, dtype=np.uint8)

        # Stack nei canali RGB
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def get_segmented_frame(self, pixel_to_mm = 10):
        ret, frame = self.cap.read()                                                            # frame [w, h, 3]
        frame = cv2.resize(frame, (640, 360))
        # ---------------- ROI ----------------
        ROI_X = 199 # 150
        ROI_Y = 59  # 80
        ROI_W = 455-ROI_X # 320
        ROI_H = 315-ROI_Y # 320
        # Assicura che la ROI sia valida
        x = max(0, ROI_X)
        y = max(0, ROI_Y)
        w = min(ROI_W, frame.shape[:2][1] - x)
        h = min(ROI_H, frame.shape[:2][0] - y)
        frame = frame[y:y+h, x:x+w]
        if frame.shape[0] != frame.shape[1]: raise ValueError(f"Frame must be squared: {frame.shape[:2]}")
        # -------------------------------------
        if not ret: return None, None
        mask = self.segmentator.get_segmented_mask(frame) * 255.0                               # mask [256, 256] uint
        frame = cv2.resize(frame, (self.IMG_W, self.IMG_H))
        vis_frame = frame.copy()
        frame = self.append_frame_and_mask(frame[:,:,0], mask)                                  # frame [256, 256, 3] => [R: frame, G: mask, B: unused]
        
        # ============================================================
        # Enclosing circle
        # ============================================================
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        num_areas = num_labels - 1  # esclude lo sfondo

        points = cv2.findNonZero(mask)
        diameter = 0.0
        if points is not None and num_areas == 1:
            (cx, cy), radius = cv2.minEnclosingCircle(points)

            center = (int(cx), int(cy))
            radius = int(radius)

            # Prendo il canale blu come array contiguo
            blue = vis_frame[:, :, 0].copy()

            # Disegno SOLO sul canale blu
            cv2.circle(blue, center, radius, 255, 1)

            # Centro opzionale
            cv2.circle(blue, center, 2, 255, -1)

            # Rimetto il canale modificato nel frame
            vis_frame[:, :, 0] = blue
            diameter = 2.0 * radius * pixel_to_mm
        # ============================================================

        self.im.set_data(vis_frame / 255.0)
        plt.pause(0.05)
        return frame, mask, diameter  

    def app_achived_result(self, frame_index, mask):
        if self.app == "AORTA":
            if mask.sum() > self.models[self.app]['MIN_SEGMENTED_PIXEL']:
                frame_index = frame_index + 1
                if frame_index >= self.models[self.app]['MIN_SUCCESS_FRAMES']:
                    return True, frame_index, mask.sum()
                else: return False, frame_index, mask.sum()
            else: return False, 0, mask.sum()
        if self.app == "JUGUL":
            pass
        if self.app == "CAROT":
            pass

    def start_app(self):
        self.liralabSocket.write("RUN")

        if self.app == "AORTA": self.start_aorta_app()
        if self.app == "JUGUL": self.start_jugular_app()
        if self.app == "CAROT": self.start_carotid_app()

    def start_aorta_app(self):
        first = True
        self.segmentator = Segmentator(self.models['AORTA']['SEG'], self.models['AORTA']['SEG_MODEL'])
        initial_time = time.time()
        while True:
            msg = self.liralabSocket.read()
            if msg == "MEASURE":
                diameters = []
                start_time = time.time()
                mean_diameter = -1

                while time.time() - start_time < 3.0:
                    frame, mask, diameter = self.get_segmented_frame(self.models['AORTA']['PIXEL_TO_MM'])
                    if frame is None: break
                    if diameter > self.models['AORTA']['MIN_DIAMETER']: diameters.append(diameter)

                if len(diameters) > self.models['AORTA']['FRAME_TO_SUCCESS']:
                    mean_diameter = float(np.mean(diameters))

                if mean_diameter > 0:
                    print(f"diametro medio: {mean_diameter:.2f} mm")
                    print(f"tempo: {time.time() - initial_time:.2f} s")
                print("Next point...")
                self.liralabSocket.write(f"{mean_diameter};")

    def start_jugular_app(self):
        pass
    
    def start_carotid_app(self):
        pass

if __name__ == "__main__":
    ilControl = liralabGRIDControl("AORTA")
    ilControl.start_app()