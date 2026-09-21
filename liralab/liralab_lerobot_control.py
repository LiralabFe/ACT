"""
Esempio minimo: carica una checkpoint di Diffusion Policy addestrata con LeRobot
e fa un'inferenza con dati finti, solo per verificare che il caricamento
e il formato degli input siano corretti.
"""
import torch
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.policies.act import ACTPolicy
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors
from liralab.utils.segmentator import Segmentator
from liralab.liralab_socket import LiralabSocket
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial.transform import Rotation as R
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class liralabLeRobotControl:
    def __init__(self):
        # parameters
        self.params ={
            'AORTA' : 
            {
                'MODEL':"/home/legion/PycharmProjects/lerobot/outputs/train/AAA_8/checkpoints/020000/pretrained_model",
                'SEG' : "/home/legion/PycharmProjects/ACT/ACT_refactor/segmentation_models/hardsmeg/hardnet68.pth",
                'SEG_MODEL' : "HarDMSEG",
                'MIN_SUCCESS_FRAMES' : 15,
                'BUFFER_FRAMES' : 100,
                'FRAME_TO_SUCCESS' : 15,
                'MIN_DIAMETER' : 9,
                'PIXEL_TO_MM' : 1.0/1.8, # 1.8 pixels = 1mm nella ROI attuale ( Zoom: 27 Hz)
            }
        }

        # ---------- LEROBOT POLICY STUFF
        self.CHECKPOINT_DIR = self.params['AORTA']['MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # self.policy = ACTPolicy.from_pretrained(self.CHECKPOINT_DIR).to(self.device)
        self.policy = DiffusionPolicy.from_pretrained(self.CHECKPOINT_DIR).to(self.device)
        #self.policy = SmolVLAPolicy.from_pretrained(self.CHECKPOINT_DIR).to(self.device)
        self.policy.eval()

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.CHECKPOINT_DIR,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        # ---------- 
        self.IMG_H = 256 # 480
        self.IMG_W = 256 # 640

        # ---------- SEGMENTATOR
        self.segmentator = None

        # ---------- INIT
        self.T_initial_0 = None
        self.T_0_initial = None

        self.use_force_sensor = True
        self.liralabSocket = LiralabSocket(5028)
        self.cap = cv2.VideoCapture("VideoAorta.mp4")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = self.cap.read()
        while frame is None or frame.max() == 0:
            ret, frame = self.cap.read()
            time.sleep(0.5)
        frame = cv2.resize(frame, (640, 360))
        print(frame.shape)
        plt.figure(figsize=(12, 7))
        plt.imshow(frame)
        plt.show()

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(16, 16))
        self.im = self.ax.imshow(np.zeros((256,256,3)))


    def run(self):
        # Register init pose
        state = self.liralabSocket.read().split(';')[:-1]
        self.T_initial_0 = self.get_tran_from_state(state)
        self.T_0_initial = np.linalg.inv(self.T_initial_0)
        self.liralabSocket.write("RUN")
        self.__run_aorta_app()

    def get_tran_from_state(self,state):
        return np.array([
        [state[3],state[4],state[5],state[0]],
        [state[6],state[7],state[8],state[1]],
        [state[9],state[10],state[11],state[2]],
        [0,0,0,1]
        ], dtype=np.float32)

    def get_current_ee_from_initial(self):
        state = self.liralabSocket.read().split(';')[:-1]
        if self.use_force_sensor:
            force = self.get_force_from_state(state)

        T_curr_0 = self.get_tran_from_state(state)                                              # T from current position to origin
        T_curr_initial = np.dot(self.T_0_initial, T_curr_0)                                     # T from current position to belly
        rpy = R.from_matrix(T_curr_initial[:3,:3]).as_euler('xyz').astype(np.float32)           # roll pitch yaw

        if self.use_force_sensor:
            ee_curr_initial = np.concatenate([T_curr_initial[:3,3], rpy, force])
        else:
            ee_curr_initial = np.concatenate([T_curr_initial[:3,3], rpy])

        return ee_curr_initial

    def get_force_from_state(self, state):
        return np.array([
            state[12], state[13], state[14]
        ], dtype=np.float32)
    
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
        plt.pause(0.01)
        return frame.astype(np.float32) / 255.0, mask.astype(np.float32) / 255.0, diameter

    def preprocess_frame(self,frame_rgb):
        # resize
        frame = cv2.resize(frame_rgb, (self.IMG_W, self.IMG_H))

        # to tensor
        frame = torch.from_numpy(frame).float()     # [H,W,C]
        frame = frame.permute(2, 0, 1)              # [C,H,W]

        return frame.unsqueeze(0)  # [1,C,H,W]
    
    def transform_to_string(self, T):
        # Estrai traslazione
        x, y, z = T[0:3, 3]

        # Estrai rotazione 3x3 (flatten riga per riga)
        R = T[0:3, 0:3].flatten()

        values = [x, y, z] + R.tolist()

        return ";".join(f"{v:.4f}" for v in values)

    def __run_aorta_app(self):
        first = True
        self.segmentator = Segmentator(self.params['AORTA']['SEG'], self.params['AORTA']['SEG_MODEL'])
        ee_new_belly_old = None
        diameters = deque(maxlen=self.params['AORTA']['BUFFER_FRAMES'])
        for i in range(10):
            ret, frame = self.cap.read()                                                            # frame [w, h, 3]

        while(True):
            #------------------------#
            # Read state from socket #
            #------------------------#
            ee_curr_belly = self.get_current_ee_from_initial()
            if first:
                first = False
                start = time.perf_counter()
                print(">>> START TIMER <<<")

            #--------------------------------#
            # Capture frame for segmentation #
            #--------------------------------#
            frame, mask, diameter = self.get_segmented_frame(self.params['AORTA']['PIXEL_TO_MM'])
            if frame is None: break

            #-------------------#
            # Success Condition #
            #-------------------#
            #diameters.append(diameter)
            #above_threshold = 0
            #mean_diameter = 0
            #for i in range(len(diameters)):
            #    if diameters[i] > self.params['AORTA']['MIN_DIAMETER']:
            #        above_threshold += 1
            #        mean_diameter += diameters[i]
            #    if above_threshold > self.params['AORTA']['FRAME_TO_SUCCESS']:
            #        print(f"MEAN DIAMETER: {mean_diameter/above_threshold:.1f}")
            #        elapsed = time.perf_counter() - start - 3.2
            #        print(f"Tempo: {elapsed:.2f} s")
            #        return
            #if above_threshold % 5 == 0 and above_threshold > 0: print(f"Above: {above_threshold}")

            frame = self.preprocess_frame(frame) # [3,256,256] float32

            observation = {
                "observation.images.top": frame,  # (1, 3, H, W) - batch di 1
                "observation.state": torch.from_numpy(ee_curr_belly).unsqueeze(0).to(self.device),             # (1, 9)
                "task": "Guide the ultrasound on the abdomen to find and center the aorta."
            }

            # ----------------------------------------------------------------------
            # 5. Inferenza
            # ----------------------------------------------------------------------
            with torch.no_grad():
                obs = self.preprocess(observation)          # normalizza e sposta su device
                action = self.policy.select_action(obs)     # (1, 6) - una azione dal chunk predetto
                ee_new_belly = self.postprocess(action)[0]           # denormalizza -> unità reali (metri, radianti)
        

            # Bounding box rotation
            limit = 5 * np.pi / 180  # ≈ 0.174532925 rad
            if ee_new_belly_old is not None:
                if(np.abs((ee_new_belly[3] - ee_new_belly_old[3]) * 180.0 / np.pi) > 10):
                    print("X: " + str((ee_new_belly[3] - ee_new_belly_old[3]) * 180.0 / np.pi))
                
                if(np.abs((ee_new_belly[4] - ee_new_belly_old[4]) * 180.0 / np.pi) > 10):
                    print("Y: " + str((ee_new_belly[4] - ee_new_belly_old[4]) * 180.0 / np.pi))

                if(np.abs((ee_new_belly[5] - ee_new_belly_old[5]) * 180.0 / np.pi) > 10):
                    print("Z: " + str((ee_new_belly[5] - ee_new_belly_old[5]) * 180.0 / np.pi) + ""
                    f" with old {ee_new_belly_old[5] * 180.0 / np.pi} and new {ee_new_belly[5] * 180.0 / np.pi}")
            ee_new_belly_old = ee_new_belly
            ee_new_belly[3] = 0.0 # np.clip(ee_new_belly[3], -limit, limit)
            ee_new_belly[4] = 0.0 # np.clip(ee_new_belly[4], -limit, limit)

            eeR = R.from_euler('xyz', ee_new_belly[3:]).as_matrix()
            eeR = np.concatenate([eeR[0],eeR[1],eeR[2]])
            eeP = ee_new_belly[0:3]
            ee_new_belly = np.concatenate([eeP,eeR])
            ee_new_belly = self.get_tran_from_state(ee_new_belly)
            ee_new_0 = np.dot(self.T_initial_0 , ee_new_belly)

            #-----------------#
            # Write new state #
            #-----------------#
            self.liralabSocket.write(self.transform_to_string(ee_new_0))
    
if __name__ == "__main__":
    ilControl = liralabLeRobotControl()
    ilControl.run()