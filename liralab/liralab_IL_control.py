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


class liralabILControl:
    def __init__(self, APP : str):
        assert APP in ['AORTA', 'JUGUL', 'CAROT'], f"{APP} not in ['AORTA', 'JUGUL', 'CAROT']"

        self.app = APP
        self.models = {
            'AORTA' : {
                'ACT' : "experiments/AAA/policy_last.ckpt",
                'SEG' : "segmentation_models/unet_dnet121_case_v1_AORTA.h5",
                'DATASET' : "data/liralab/AAA",
                'MIN_SEGMENTED_PIXEL' : 100,
                'MIN_SUCCESS_FRAMES' : 40,
            },
            'JUGUL' : {
                'ACT' : "experiments/JVP/policy_last.ckpt",
                'SEG' : "segmentation_models/unet_dnet121_case_v1_NECK.h5",
                'DATASET' : "data/liralab/JVP",
            },
            'CAROT' : {
                'ACT' : "experiments/CAS/policy_last.ckpt",
                'SEG' : "segmentation_models/unet_dnet121_case_v1_NECK.h5",
                'DATASET' : "data/liralab/CAS",
            },
        }

        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus:
            tf.config.set_visible_devices([], 'GPU')
        # --------- NEURAL NETWORK
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        self.policy = ACTPolicy()
        self.policy.cuda()
        self.policy.load_state_dict(torch.load(self.models[APP]["ACT"]))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy.eval()

        # ---------- NORMALIZATION
        self.IMG_H = 256 # 480
        self.IMG_W = 256 # 640
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],    std=[0.229, 0.224, 0.225])
        self.dataset_stats = get_norm_stats(self.models[APP]['DATASET'])
        self.qpos_mean = np.array(self.dataset_stats['qpos_mean'], dtype=np.float32)
        self.qpos_std = np.array(self.dataset_stats['qpos_std'], dtype=np.float32)
        self.T_initial_0 = None
        self.T_0_initial = None

        # ---------- INIT
        self.liralabSocket = LiralabSocket(5002)
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        while frame.max() == 0:
            ret, frame = self.cap.read()
            time.sleep(0.5)
        plt.imshow(frame)
        plt.show()
        self.seg_model = tf.keras.models.load_model(self.models[APP]['SEG'],compile=False)

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

    def get_segmented_frame(self):
        ret, frame = self.cap.read()                                                            # frame [w, h, 3]
        if not ret: return None, None
        frame = cv2.resize(frame, (self.IMG_W, self.IMG_H), interpolation=cv2.INTER_LINEAR)     # frame [255, 255, 3]
        frame = np.expand_dims(frame, axis=0)                                                   # frame [1, 255, 255, 3]
        mask = self.seg_model.predict(frame, verbose=1)[0]                                      # mask  [1, 255, 255]
        mask = (mask > 0.5).astype(np.float32).squeeze()*255.0                                  # mask  [255, 255]
        frame = frame.squeeze()[:,:,0].squeeze()                                                # frame [255, 255]
        frame = self.append_frame_and_mask(frame, mask)                                         # frame [255, 255, 3] => [R: frame, G: mask, B: unused]
        return frame, mask                                                                      # uint8, uint8 [0-255]

    def get_current_ee_from_initial(self):
        state = self.liralabSocket.read().split(';')
        T_curr_0 = self.get_tran_from_state(state)                                              # T from current position to origin
        T_curr_initial = np.dot(self.T_0_initial, T_curr_0)                                     # T from current position to belly
        rpy = R.from_matrix(T_curr_initial[:3,:3]).as_euler('xyz').astype(np.float32)           # roll pitch yaw
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
        frame_index = 0
        while(True):
            #------------------------#
            # Read state from socket #
            #------------------------#
            ee_curr_belly = self.get_current_ee_from_initial()

            #--------------------------------#
            # Capture frame for segmentation #
            #--------------------------------#
            frame, mask = self.get_segmented_frame()
            if frame is None: break
            result, frame_index, aorta_pixels = self.app_achived_result(frame_index, mask)
            print(f"FRAME: {frame_index} -- {mask.sum()}")
            if result: return # ---- SUCCESS ----

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
            ee_new_belly = ACT_output_action.cpu().squeeze()[0].detach().numpy() * self.qpos_std + self.qpos_mean
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