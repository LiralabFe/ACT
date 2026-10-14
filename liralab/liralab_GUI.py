"""GUI Liralab. Avvio: python -m liralab.liralab_GUI.

update_image accetta immagini PIL o array NumPy RGB (per OpenCV convertire
prima da BGR a RGB). update_image e log sono utilizzabili anche da thread
di elaborazione; caricamento e inferenza vengono eseguiti in background.
"""

import tkinter as tk
from datetime import datetime
from queue import Empty, Queue
from threading import Lock, Thread
from tkinter import scrolledtext, ttk

from PIL import Image, ImageTk


import torch
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.policies.act import ACTPolicy
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors
from liralab.utils.segmentator import Segmentator
from liralab.liralab_socket import LiralabSocket
from liralab.lowpass_filter import LowPassFilter
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial.transform import Rotation as R
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import onnxruntime as ort
import csv

class LiralabGUIControl:
    def __init__(self, gui):
        self.gui = gui

        example_image = Image.open("/home/legion/PycharmProjects/ACT/ACT_refactor/DETR.png")
        self.gui.update_image(example_image)
        # Per un frame OpenCV (BGR):
        # self.gui.update_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # parameters
        self.params ={
            'AORTA' : 
            {
                'MODEL':"/home/legion/PycharmProjects/lerobot/outputs/train/AAA_13/checkpoints/100000/pretrained_model",
                'SEG' : "/home/legion/PycharmProjects/ACT/ACT_refactor/segmentation_models/hardsmeg/HarD-MSEG-best_PHANTOM_v2.pth",
                'SEG_MODEL' : "HarDMSEG",
                'MIN_SUCCESS_FRAMES' : 15,
                'BUFFER_FRAMES' : 100,
                'FRAME_TO_SUCCESS' : 10,
                'MIN_DIAMETER' : 9,
                'PIXEL_TO_MM' : 1.0/1.8, # 1.8 pixels = 1mm nella ROI attuale ( Zoom: 27 Hz)
            },
            'JUGULAR':
            {
                'MODEL' : "/home/legion/PycharmProjects/ACT/ACT_refactor/segmentation_models/jugular/SeResNet50_Phantom(Neck)_v4.onnx",
                'VIDEO' : "/home/legion/Downloads/jugular.mp4"
            },
            'FORCE_CONTROL':True
        }
        self.liralabSocket = LiralabSocket(5025)
        self.should_quit = False
        # ---------- 
        self.IMG_H = 256 # 480
        self.IMG_W = 256 # 640

        # ---------- SEGMENTATOR
        self.segmentator = None

        # ---------- INIT
        self.lowpass = LowPassFilter(0.5)
        self.T_initial_0 = None
        self.T_0_initial = None

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = self.cap.read()
        while frame is None or frame.max() == 0:
            ret, frame = self.cap.read()
            time.sleep(0.5)
        frame = cv2.resize(frame, (640, 360))

        self.gui.update_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        

    def run_aorta(self):
        self.liralabSocket.write("AORTA")
        # ---------- LEROBOT POLICY STUFF
        self.CHECKPOINT_DIR = self.params['AORTA']['MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.policy = ACTPolicy.from_pretrained(self.CHECKPOINT_DIR).to(self.device)
        #self.policy = DiffusionPolicy.from_pretrained(self.CHECKPOINT_DIR).to(self.device)
        #self.policy = SmolVLAPolicy.from_pretrained(self.CHECKPOINT_DIR).to(self.device)
        self.policy.eval()

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.CHECKPOINT_DIR,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )


    def get_tran_from_state(self,state):
        return np.array([
        [state[3],state[4],state[5],state[0]],
        [state[6],state[7],state[8],state[1]],
        [state[9],state[10],state[11],state[2]],
        [0,0,0,1]
        ], dtype=np.float32)

    def get_current_ee_from_initial(self):
        state = self.liralabSocket.read().split(';')[:-1]

        force = self.get_force_from_state(state)

        T_curr_0 = self.get_tran_from_state(state)                                              # T from current position to origin
        T_curr_initial = np.dot(self.T_0_initial, T_curr_0)                                     # T from current position to belly
        rpy = R.from_matrix(T_curr_initial[:3,:3]).as_euler('xyz').astype(np.float32)           # roll pitch yaw

        if not self.params['FORCE_CONTROL']:
            ee_curr_initial = np.concatenate([T_curr_initial[:3,3], rpy, force])
        else:
            ee_curr_initial = np.concatenate([T_curr_initial[:3,3], force])

        return ee_curr_initial

    def get_force_from_state(self, state):
        return np.array([state[12], state[13], state[14]], dtype=np.float32)
    
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
        if not ret or frame is None:
            return None, None, None
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

        vis_frame = np.clip(vis_frame, 0, 255).astype(np.uint8)
        self.gui.update_image(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
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

    def force_control_to_string(self, T, zForce):
        x, y, z = T[0:3, 3]
        values = [x, y, z, zForce]
        return ";".join(f"{v:.4f}" for v in values)

    def __run_aorta_force_control_app(self):
        self.liralabSocket.write("START")
        # Register init pose
        state = self.liralabSocket.read().split(';')[:-1]
        self.T_initial_0 = self.get_tran_from_state(state)
        self.T_0_initial = np.linalg.inv(self.T_initial_0)
        self.liralabSocket.write("RUN")

        first = True
        self.segmentator = Segmentator(self.params['AORTA']['SEG'], self.params['AORTA']['SEG_MODEL'])
        diameters = deque(maxlen=self.params['AORTA']['BUFFER_FRAMES'])
        for i in range(10):
            ret, frame = self.cap.read()                                                            # frame [w, h, 3]

        above_threshold = 0
        mean_diameter = 0
        while(True):
            if self.should_quit:
                self.liralabSocket.write("QUIT")
                return
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

            if diameter > self.params['AORTA']['MIN_DIAMETER']:
                above_threshold += 1
                mean_diameter += diameter
                if above_threshold >= self.params['AORTA']['FRAME_TO_SUCCESS']:
                    print(f"MEAN DIAMETER: {mean_diameter/above_threshold:.1f}")
                    elapsed = time.perf_counter() - start - 3.2
                    print(f"Tempo: {elapsed:.2f} s")
                    self.liralabSocket.write("QUIT")
                    return
                    
            #for i in range(len(diameters)):
            #    if diameters[i] > self.params['AORTA']['MIN_DIAMETER']:
            #        above_threshold += 1
            #        mean_diameter += diameters[i]
            #    if above_threshold > self.params['AORTA']['FRAME_TO_SUCCESS']:
            #        print(f"MEAN DIAMETER: {mean_diameter/above_threshold:.1f}")
            #        elapsed = time.perf_counter() - start - 3.2
            #        print(f"Tempo: {elapsed:.2f} s")
            #        return
            if above_threshold % 5 == 0 and above_threshold > 0: print(f"Above: {above_threshold}")

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
                action = self.policy.select_action(obs)     # (1, 4) - una azione dal chunk predetto
                action = self.postprocess(action)[0]           # denormalizza -> unità reali (metri, radianti)
        

            force_limit = 12 # N
            action[3] = np.clip(action[3], -force_limit, force_limit)
            action[3] = self.lowpass.update(action[3])
            print(f"FORCE: {action[3]}")
            eeR = np.eye(3)
            eeR = np.concatenate([eeR[0],eeR[1],eeR[2]])
            eeP = action[0:3]
            ee_new_belly = np.concatenate([eeP,eeR])
            ee_new_belly = self.get_tran_from_state(ee_new_belly)
            ee_new_0 = np.dot(self.T_initial_0 , ee_new_belly)

            #-----------------#
            # Write new state #
            #-----------------#
            self.liralabSocket.write(self.force_control_to_string(ee_new_0,action[3]))


class LiralabGUI(tk.Tk):
    def __init__(self, on_start=None):
        """on_start, se fornito, riceve la modalità scelta come stringa."""
        super().__init__()
        self.title("Liralab")
        self.geometry("900x700")
        self.minsize(500, 400)
        self.on_start = on_start
        self.mode = tk.StringVar(value="AORTA")
        self._events = Queue()
        self._image_lock = Lock()
        self._pending_image = None
        self._busy = False
        self._ready = False
        self._mode_selected = False
        self._quit_requested = False
        self.liralab_GUI_control = None
        self._image = None
        self._photo = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        controls = ttk.Frame(self, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        self._mode_buttons = {}
        for column, mode in enumerate(("AORTA", "JUGULAR")):
            button = ttk.Button(controls, text=mode, command=lambda value=mode: self.select_mode(value))
            button.grid(row=0, column=column, padx=(0, 8))
            self._mode_buttons[mode] = button
        self.start_button = ttk.Button(controls, text="START", command=self.start)
        self.start_button.grid(row=0, column=2)
        self.start_button.state(["disabled"])
        self.quit_button = ttk.Button(controls, text="QUIT", command=self.request_quit)
        self.quit_button.grid(row=0, column=3, padx=(8, 0))
        self.quit_button.state(["disabled"])
        ttk.Label(controls, textvariable=self.mode).grid(row=0, column=4, padx=16)
        #self._mode_buttons["AORTA"].state(["pressed"])

        image_frame = ttk.LabelFrame(self, text="Immagine", padding=8)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(image_frame, background="#20252b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._render_image)

        log_frame = ttk.LabelFrame(self, text="LOG", padding=8)
        log_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        self.log_text = scrolledtext.ScrolledText(log_frame, height=7, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.log("Interfaccia pronta. Modalità: AORTA.")
        self.after(40, self._process_events)
        self._run_background("init", self._initialize_controller)

    def _initialize_controller(self):
        self.liralab_GUI_control = LiralabGUIControl(self)

    def _run_background(self, operation, task):
        if self._busy:
            return
        self._busy = True
        self.start_button.state(["disabled"])
        self.quit_button.state(["!disabled"] if operation == "run" else ["disabled"])
        for button in self._mode_buttons.values():
            button.state(["disabled"])

        def work():
            success = False
            try:
                task()
                success = True
            except Exception as exc:
                self.log(f"Errore ({operation}): {exc}")
            finally:
                # Solo il thread GUI modifica lo stato dei widget.
                self._events.put(("done", (operation, success)))

        Thread(target=work, name=f"liralab-{operation}", daemon=True).start()

    def select_mode(self, mode):
        if self._busy or self._mode_selected or self.liralab_GUI_control is None:
            return
        self._mode_selected = True
        self._quit_requested = False
        self.liralab_GUI_control.should_quit = False
        self._ready = False
        self.mode.set(mode)
        for name, button in self._mode_buttons.items():
            button.state(["pressed"] if name == mode else ["!pressed"])

        self.log(f"Modalità selezionata: {mode}")
        def prepare():
            getattr(self.liralab_GUI_control, f"run_{mode.lower()}")()

        self._run_background("prepare", prepare)

    def start(self):
        if self._busy or not self._ready:
            return
        mode = self.mode.get()
        self.log(f"START: {mode}")
        if self.on_start is not None:
            self._run_background("run", lambda: self.on_start(mode))
        elif mode == "AORTA":
            self._run_background(
                "run", self.liralab_GUI_control._LiralabGUIControl__run_aorta_force_control_app
            )
        else:
            self.log("Collegare on_start per avviare l'elaborazione.")

    def request_quit(self):
        self._quit_requested = True
        self._mode_selected = False
        if self.liralab_GUI_control is not None:
            self.liralab_GUI_control.should_quit = True
            self.log("QUIT richiesto.")
        for button in self._mode_buttons.values():
            button.state(["!disabled", "!pressed"])
        self.start_button.state(["!disabled"])
        self.quit_button.state(["disabled"])

    def update_image(self, image):
        """Mostra una copia dell'immagine, mantenendone le proporzioni."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.copy()
        with self._image_lock:
            self._pending_image = image

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._events.put(("log", f"[{timestamp}] {message}\n"))

    def _process_events(self):
        for _ in range(100):
            try:
                kind, value = self._events.get_nowait()
            except Empty:
                break
            if kind == "done":
                operation, success = value
                self._busy = False
                self.quit_button.state(["disabled"])
                if operation == "prepare":
                    self._ready = success
                elif operation == "run":
                    self._ready = False
                    if success:
                        self._mode_selected = False
                for button in self._mode_buttons.values():
                    button.state(["disabled"] if self._mode_selected else ["!disabled"])
                self.start_button.state(
                    ["!disabled"]
                    if self._ready or self._quit_requested or (operation == "run" and success)
                    else ["disabled"]
                )
                if success:
                    self.log(f"Completato: {operation}")
            else:
                self.log_text.configure(state="normal")
                self.log_text.insert("end", value)
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        with self._image_lock:
            image = self._pending_image
            self._pending_image = None
        if image is not None:
            self._image = image
            self._render_image()
        self.after(40, self._process_events)

    def _render_image(self, event=None):
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete("all")
        if self._image is None:
            self.canvas.create_text(
                width / 2, height / 2, text="In attesa di un'immagine", fill="white"
            )
            return
        preview = self._image.copy()
        preview.thumbnail((width, height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(width / 2, height / 2, image=self._photo)


if __name__ == "__main__":
    LiralabGUI().mainloop()
