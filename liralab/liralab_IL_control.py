import liralab.model
from liralab.model import *
import numpy as np
from spatialmath import SE3
import cv2
import torchvision.transforms as transforms
# import roboticstoolbox as rtb
import numpy as np
import socket
import struct
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

class liralabILControl:
    def __init__(self, APP : str):
        assert APP in ['AORTA', 'JUGUL', 'CAROT'], f"{APP} not in ['AORTA', 'JUGUL', 'CAROT']"

        self.use_force_sensor = True
        self.app = APP
        self.models = {
            'AORTA' : {
                'ACT' : "experiments/AAA_22/policy_epoch_8078.ckpt",
                'SEG' : "/home/legion/PycharmProjects/ACT/ACT_refactor/segmentation_models/hardsmeg/hardnet68.pth",
                'SEG_MODEL' : "HarDMSEG",
                'MIN_SUCCESS_FRAMES' : 40,
                'BUFFER_FRAMES' : 100,
                'FRAME_TO_SUCCESS' : 40,
                'MIN_DIAMETER' : 7,
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

        # ---------- ARGS FROM JSON
        args_path = Path(self.models[self.app]["ACT"]).parent / "args.json"
        with open(args_path, "r") as f:
            args = json.load(f)
            liralab.model.args = args

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = ACTPolicy().to(self.device)
        self.policy.load_state_dict(torch.load(self.models[APP]["ACT"], map_location=self.device))
        self.policy.eval()

        # ---------- SEGMENTATOR
        self.segmentator = None

        # ---------- NORMALIZATION
        self.IMG_H = 256 # 480
        self.IMG_W = 256 # 640
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std)
        self.seg_normalization = A.Compose([A.Normalize(mean, std),A.ToFloat(max_value=255.0),ToTensorV2()])
        self.dataset_stats = self.get_norm_stat(args)

        self.qpos_mean = np.array(self.dataset_stats['qpos_mean'], dtype=np.float32)
        self.qpos_std = np.array(self.dataset_stats['qpos_std'], dtype=np.float32)
        self.T_initial_0 = None
        self.T_0_initial = None

        # ---------- INIT
        self.liralabSocket = LiralabSocket(5000)
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        while frame.max() == 0:
            ret, frame = self.cap.read()
            time.sleep(0.5)
        print(frame.shape)
        plt.imshow(frame)
        plt.show()

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.zeros_like(frame))

    def get_norm_stat(self, args):
        # New arsg.jsons have dataset' stats in it, otherwise recalculate them from the actual dataset
        if 'dataset_stats' in args and args['dataset_stats'] is not None:
            return {
                "action_mean" : np.array(args['dataset_stats']["action_mean"]),
                "action_std" : np.array(args['dataset_stats']["action_std"]),
                "qpos_mean" : np.array(args['dataset_stats']["qpos_mean"]),
                "qpos_std" : np.array(args['dataset_stats']["qpos_std"])
            }
        else:
            return get_norm_stats(args['dataset_dir'])

    def preprocess_frame(self,frame_bgr):
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # resize
        frame_rgb = cv2.resize(frame_rgb, (self.IMG_W, self.IMG_H))

        # to tensor
        frame = torch.from_numpy(frame_rgb).float()  # [H,W,C]
        frame = frame.permute(2, 0, 1)               # [C,H,W]

        frame /= 255.0
        frame = self.normalize(frame)
        
        return frame.unsqueeze(0)  # [1,C,H,W]

    def get_tran_from_state(self,state):
        return np.array([
        [state[3],state[4],state[5],state[0]],
        [state[6],state[7],state[8],state[1]],
        [state[9],state[10],state[11],state[2]],
        [0,0,0,1]
        ], dtype=np.float32)
    
    def get_force_from_state(self, state):
        return np.array([
            state[12], state[13], state[14]
        ], dtype=np.float32)

    def append_frame_and_mask(self, frame, mask):
        # Converti in numpy array
        r = np.array(frame, dtype=np.uint8)
        g = np.array(mask, dtype=np.uint8)
        b = np.zeros_like(r, dtype=np.uint8)

        # Stack nei canali RGB
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def transform_to_string(self, T):
        # Estrai traslazione
        x, y, z = T[0:3, 3]

        # Estrai rotazione 3x3 (flatten riga per riga)
        R = T[0:3, 0:3].flatten()

        values = [x, y, z] + R.tolist()

        return ";".join(f"{v:.4f}" for v in values)

    def get_segmented_frame(self, pixel_to_mm = 10):
        ret, frame = self.cap.read()                                                            # frame [w, h, 3]
        # ---------------- ROI ----------------
        ROI_X = 150
        ROI_Y = 80
        ROI_W = 320
        ROI_H = 320
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
        points = cv2.findNonZero(mask)
        diameter = 0.0
        if points is not None:
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
            print(vis_frame.shape)
            diameter = 2.0 * radius * pixel_to_mm
        # ============================================================

        self.im.set_data(vis_frame / 255.0)
        plt.pause(0.05)
        return frame, mask, diameter  

    def get_current_ee_from_initial(self):
        state = self.liralabSocket.read().split(';')
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
        # Register init pose
        state = self.liralabSocket.read().split(';')[:-1]
        self.T_initial_0 = self.get_tran_from_state(state)
        self.T_0_initial = np.linalg.inv(self.T_initial_0)
        self.liralabSocket.write("RUN")

        if self.app == "AORTA": self.start_aorta_app()
        if self.app == "JUGUL": self.start_jugular_app()
        if self.app == "CAROT": self.start_carotid_app()

    def start_aorta_app(self):
        self.segmentator = Segmentator(self.models['AORTA']['SEG'], self.models['AORTA']['SEG_MODEL'])
        ee_new_belly_old = None
        diameters = deque(maxlen=self.models['AORTA']['BUFFER_FRAMES'])
        while(True):
            #------------------------#
            # Read state from socket #
            #------------------------#
            ee_curr_belly = self.get_current_ee_from_initial()

            #--------------------------------#
            # Capture frame for segmentation #
            #--------------------------------#
            frame, mask, diameter = self.get_segmented_frame(self.models['AORTA']['PIXEL_TO_MM'])
            if frame is None: break

            #-------------------#
            # Success Condition #
            #-------------------#
            diameters.append(diameter)
            above_threshold = 0
            mean_diameter = 0
            print(f"Current diameter {diameter}")
            for i in range(len(diameters)):
                if diameters[i] > self.models['AORTA']['MIN_DIAMETER']:
                    above_threshold += 1
                    mean_diameter += diameters[i]
                if above_threshold > self.models['AORTA']['FRAME_TO_SUCCESS']:
                    print(f"MEAN DIAMETER: {mean_diameter/above_threshold}")
                    return
            print(f"Above: {above_threshold}")

            #-------------------------#
            # Normalize input for ACT #
            #-------------------------#
            ACT_input_state = (ee_curr_belly - self.qpos_mean) / self.qpos_std
            ACT_input_state = torch.from_numpy(ACT_input_state).unsqueeze(0).to(self.device)
            ACT_input_image = self.preprocess_frame(frame).to(self.device)

            #---------------#
            # ACT Inference #
            #---------------#
            ACT_output_action = self.policy(ACT_input_state, ACT_input_image)

            #--------------------#
            # Denormalize output #
            #--------------------#
            ee_new_belly = ACT_output_action.cpu().squeeze()[0].detach().numpy() * self.qpos_std[:6] + self.qpos_mean[:6]

            # Bounding box rotation
            limit = 10 * np.pi / 180  # ≈ 0.174532925 rad
            if ee_new_belly_old is not None:
                if(np.abs((ee_new_belly[3] - ee_new_belly_old[3]) * 180.0 / np.pi) > 10):
                    print("X: " + str((ee_new_belly[3] - ee_new_belly_old[3]) * 180.0 / np.pi))
                
                if(np.abs((ee_new_belly[4] - ee_new_belly_old[4]) * 180.0 / np.pi) > 10):
                    print("Y: " + str((ee_new_belly[4] - ee_new_belly_old[4]) * 180.0 / np.pi))

                if(np.abs((ee_new_belly[5] - ee_new_belly_old[5]) * 180.0 / np.pi) > 10):
                    print("Z: " + str((ee_new_belly[5] - ee_new_belly_old[5]) * 180.0 / np.pi) + ""
                    f" with old {ee_new_belly_old[5] * 180.0 / np.pi} and new {ee_new_belly[5] * 180.0 / np.pi}")
            ee_new_belly_old = ee_new_belly
            ee_new_belly[3] = np.clip(ee_new_belly[3], -limit, limit)
            ee_new_belly[4] = np.clip(ee_new_belly[4], -limit, limit)

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

    def start_jugular_app(self):
        pass
    
    def start_carotid_app(self):
        pass

if __name__ == "__main__":
    ilControl = liralabILControl("AORTA")
    ilControl.start_app()