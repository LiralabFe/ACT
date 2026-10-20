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

# Colori personalizzabili (nomi Tk oppure valori esadecimali #RRGGBB).
GUI_COLORS = {
    "background": "#22141D",
    "text": "#e7bae3",
    "button": "#0B2336",
    "button_hover": "#123958",
    "button_pressed": "#4677AF",
    "button_disabled": "#141414",
    "text_disabled": "#35272f",
    "canvas": "#0F0B0E",
    "placeholder": "#22141D",
    "log_background": "#110C0F",
    "log_text": "#ade9d0",
    "selection": "#000000",
    "border": "#1B0A11"
}

class LiralabGUIControl:
    def __init__(self, gui):
        self.gui = gui

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
                'VIDEO' : "/home/legion/Downloads/jugular.mp4",
                'SEG' : "/home/legion/PycharmProjects/ACT/ACT_refactor/segmentation_models/jugular/SeResNet50_Phantom(Neck)_v4.pth",
                'SEG_MODEL' : "SeResNet50",
            },
            'FORCE_CONTROL':True
        }
        self.liralabSocket = LiralabSocket(5026)
        self.should_quit = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.cap = None
        self.jugular_datas = []

    def run_aorta(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        ret, frame = self.cap.read()
        while frame is None or frame.max() == 0:
            ret, frame = self.cap.read()
            time.sleep(0.5)
        frame = cv2.resize(frame, (640, 360))

        self.gui.update_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.liralabSocket.write("AORTA")

    def run_jugular(self):
        self.jugular_datas = []
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(self.params['JUGULAR']['VIDEO'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 10)

        ret, frame = self.cap.read()
        while frame is None or frame.max() == 0:
            ret, frame = self.cap.read()
            time.sleep(0.5)
        frame = cv2.resize(frame, (640, 360))

        self.gui.update_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.liralabSocket.write("JUGULAR")

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
            mask_3ch = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            vis_frame = cv2.addWeighted(vis_frame, 0.7, mask_3ch, 0.3, 0)
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

    def __run_aorta_grid_control_app(self):
        self.liralabSocket.write("START")

        self.segmentator = Segmentator(self.params['AORTA']['SEG'], self.params['AORTA']['SEG_MODEL'])
        diameters = deque(maxlen=self.params['AORTA']['BUFFER_FRAMES'])
        for i in range(10):
            ret, frame = self.cap.read()                                                            # frame [w, h, 3]            

        initial_time = time.time()
        while(True):
            msg = self.liralabSocket.read()
            if msg == "MEASURE":
                diameters = []
                start_time = time.time()
                mean_diameter = -1

                while time.time() - start_time < 3.0:
                    frame, mask, diameter = self.get_segmented_frame(self.params['AORTA']['PIXEL_TO_MM'])
                    if frame is None: break
                    if diameter > self.params['AORTA']['MIN_DIAMETER']: diameters.append(diameter)

                if len(diameters) > self.params['AORTA']['FRAME_TO_SUCCESS']:
                    mean_diameter = float(np.mean(diameters))

                if mean_diameter > 0:
                    print(f"diametro medio: {mean_diameter:.2f} mm")
                    print(f"tempo: {time.time() - initial_time:.2f} s")
                    self.liralabSocket.write(f"{mean_diameter};")
                    self.gui.popup(f"Diameter: {mean_diameter}\nTime: {time.time() - initial_time}.")
                    return
                print("Next point...")
                
                if self.should_quit:
                    self.liralabSocket.write("QUIT")
                    return
                self.liralabSocket.write(f"{mean_diameter};")
    
    def __run_jugular_grid_control_app(self):
        self.liralabSocket.write("START")

        self.segmentator = Segmentator(self.params['JUGULAR']['SEG'], self.params['JUGULAR']['SEG_MODEL'])
        for i in range(10):
            ret, frame = self.cap.read()                                                            # frame [w, h, 3]            

        initial_time = time.time()
        while(True):
            msg = self.liralabSocket.read()
            if msg == "MEASURE":
                start_time = time.time()
                mean_diameter = -1

                for i in range(10): # 10 seconds
                    data = []
                    for n in range (10): # 10 samples per second
                        if self.should_quit:
                            self.liralabSocket.write("QUIT")
                            return
                        time.sleep(0.1)
                        frame, mask, diameter = self.get_segmented_frame(self.params['AORTA']['PIXEL_TO_MM'])
                        if frame is None or diameter is None:
                            data.append(data[-1] if len(data) > 0 else 0)
                            continue
                        data.append(diameter)

                    self.jugular_datas.append(data)
                    self.gui.updateRealtimeGraph(np.mean(self.jugular_datas, axis=0))

                self.liralabSocket.write(f"{1.0};")
                return

class LiralabGUI(tk.Tk):
    def __init__(self, on_start=None, colors=None):
        """colors può sovrascrivere le voci di GUI_COLORS.

        on_start, se fornito, riceve la modalità scelta come stringa.
        """
        super().__init__()
        self.colors = {**GUI_COLORS, **(colors or {})}
        self._style = ttk.Style(self)
        # Il tema clam permette di impostare i colori anche sui pulsanti ttk.
        self._style.theme_use("clam")
        self.title("Liralab")
        self.geometry("900x700")
        self.minsize(500, 400)
        self.on_start = on_start
        self.mode = tk.StringVar(value="AORTA")
        self._events = Queue()
        self._image_lock = Lock()
        self._pending_image = None
        self._graph_lock = Lock()
        self._pending_graph = None
        self._graph_window = None
        self._graph_canvas = None
        self._graph_axes = None
        self._graph_line = None
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
        # La scrollbar Tk nativa può mantenere bordi di sistema chiari.
        self.log_text.vbar.destroy()
        self.log_text.vbar = ttk.Scrollbar(
            self.log_text.frame, orient="vertical", command=self.log_text.yview
        )
        self.log_text.vbar.pack(side="right", fill="y", before=self.log_text._w)
        self.log_text.configure(yscrollcommand=self.log_text.vbar.set)
        self.set_colors()
        self.log("Interfaccia pronta. Modalità: AORTA.")
        self.after(40, self._process_events)
        self._run_background("init", self._initialize_controller)

    def _initialize_controller(self):
        self.liralab_GUI_control = LiralabGUIControl(self)

    def set_colors(self, colors=None):
        """Applica i colori; chiamare dal thread GUI per cambiarli a runtime."""
        self.colors.update(colors or {})
        c = self.colors
        self.configure(background=c["background"])
        self._style.configure(
            ".", background=c["background"], foreground=c["text"],
            bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
            focuscolor=c["border"], troughcolor=c["background"],
        )
        for name in ("TFrame", "TLabel", "TLabelframe", "TLabelframe.Label"):
            self._style.configure(
                name, background=c["background"], foreground=c["text"],
                bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
                relief="flat",
            )
        self._style.configure(
            "TButton", background=c["button"], foreground=c["text"], relief="flat",
            bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
            focuscolor=c["border"],
        )
        self._style.map(
            "TButton",
            background=[("disabled", c["button_disabled"]),
                        ("pressed", c["button_pressed"]), ("active", c["button_hover"])],
            foreground=[("disabled", c["text_disabled"])],
            bordercolor=[("!disabled", c["border"]), ("disabled", c["border"])],
            lightcolor=[("!disabled", c["border"]), ("disabled", c["border"])],
            darkcolor=[("!disabled", c["border"]), ("disabled", c["border"])],
        )
        self.canvas.configure(
            background=c["canvas"],
            borderwidth=0,
            highlightthickness=0,
            relief="flat"
        )

        self.log_text.configure(
            background=c["log_background"],
            foreground=c["log_text"],
            insertbackground=c["log_text"],
            selectbackground=c["selection"],
            selectforeground=c["text"],
            borderwidth=0,
            highlightthickness=0,
            relief="flat"
        )
        self.log_text.frame.configure(
            background=c["log_background"], borderwidth=0, highlightthickness=0,
        )
        self._style.configure(
            "Vertical.TScrollbar", background=c["button"],
            troughcolor=c["log_background"], arrowcolor=c["text"],
            bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
            relief="flat",
        )
        self._style.map(
            "Vertical.TScrollbar",
            background=[("pressed", c["button_pressed"]), ("active", c["button_hover"])],
            bordercolor=[("!disabled", c["border"]), ("disabled", c["border"])],
            lightcolor=[("!disabled", c["border"]), ("disabled", c["border"])],
            darkcolor=[("!disabled", c["border"]), ("disabled", c["border"])],
        )
        
        self._render_image()

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
                "run", self.liralab_GUI_control._LiralabGUIControl__run_aorta_grid_control_app
            )
        elif mode == "JUGULAR":
            self._run_background(
                "run", self.liralab_GUI_control._LiralabGUIControl__run_jugular_grid_control_app
            )
        else:
            self.log("Collegare on_start per avviare l'elaborazione.")

    def request_quit(self):
        self._quit_requested = True
        self._ready = False
        self._mode_selected = False
        if self.liralab_GUI_control is not None:
            self.liralab_GUI_control.should_quit = True
            self.log("QUIT richiesto.")
        for button in self._mode_buttons.values():
            button.state(["!disabled", "!pressed"])
        self.start_button.state(["disabled"])
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

    def popup(self, message, title="Liralab"):
        """Mostra un messaggio non modale, anche da un thread di elaborazione."""
        self._events.put(("popup", (str(message), str(title))))

    def updateRealtimeGraph(self, newValues):
        """Aggiorna la curva Y rispetto agli indici, anche da thread del controller.

        Ogni vettore sostituisce il precedente. Un vettore vuoto cancella la curva.
        """
        values = np.array(newValues, dtype=np.float64, copy=True)
        if values.ndim != 1:
            raise ValueError("newValues deve essere un vettore monodimensionale")
        values[~np.isfinite(values)] = np.nan
        with self._graph_lock:
            self._pending_graph = values

    def _update_realtime_graph(self, values):
        if self._graph_window is None:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            window = tk.Toplevel(self)
            self._graph_window = window
            window.title("Liralab — Grafico realtime")
            window.geometry("800x450")
            window.minsize(400, 250)
            window.configure(background=self.colors["background"])
            window.transient(self)
            window.protocol("WM_DELETE_WINDOW", self._close_realtime_graph)
            window.bind("<Escape>", lambda event: self._close_realtime_graph())
            figure = Figure(figsize=(8, 4), dpi=100, layout="constrained")
            self._graph_axes = figure.add_subplot(111)
            self._graph_line, = self._graph_axes.plot([], [], linewidth=2)
            self._graph_axes.set_xlabel("Indice")
            self._graph_axes.set_ylabel("Valore")
            self._graph_axes.grid(True, alpha=0.25)
            self._graph_canvas = FigureCanvasTkAgg(figure, master=window)
            widget = self._graph_canvas.get_tk_widget()
            widget.configure(borderwidth=0, highlightthickness=0)
            widget.pack(fill="both", expand=True)

        axes = self._graph_axes
        axes.figure.set_facecolor(self.colors["background"])
        axes.set_facecolor(self.colors["canvas"])
        axes.tick_params(colors=self.colors["text"])
        axes.xaxis.label.set_color(self.colors["text"])
        axes.yaxis.label.set_color(self.colors["text"])
        for spine in axes.spines.values():
            spine.set_color(self.colors["border"])
        self._graph_line.set_color(self.colors["log_text"])
        self._graph_line.set_data(np.arange(values.size), values)
        axes.relim()
        axes.autoscale_view()
        self._graph_canvas.draw_idle()

    def _close_realtime_graph(self):
        if self._graph_window is not None:
            self._graph_window.destroy()
        self._graph_window = None
        self._graph_canvas = None
        self._graph_axes = None
        self._graph_line = None

    def _show_popup(self, message, title):
        window = tk.Toplevel(self)
        window.title(title)
        window.configure(background=self.colors["background"])
        window.transient(self)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        label = ttk.Label(
            window, text=message, font=("TkDefaultFont", 22, "bold"),
            anchor="center", justify="center", wraplength=560, padding=24,
        )
        label.grid(row=0, column=0, sticky="nsew")
        ttk.Button(window, text="OK", command=window.destroy).grid(row=1, column=0, pady=(0, 20))
        window.bind("<Escape>", lambda event: window.destroy())
        window.update_idletasks()
        width = max(420, window.winfo_reqwidth())
        height = max(200, window.winfo_reqheight())
        x = max(0, self.winfo_rootx() + (self.winfo_width() - width) // 2)
        y = max(0, self.winfo_rooty() + (self.winfo_height() - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.lift()

    def _process_events(self):
        for _ in range(100):
            try:
                kind, value = self._events.get_nowait()
            except Empty:
                break
            if kind == "popup":
                self._show_popup(*value)
            elif kind == "done":
                operation, success = value
                self._busy = False
                self.quit_button.state(["disabled"])
                if operation == "prepare":
                    self._ready = success
                elif operation == "run":
                    self._ready = False
                    self._mode_selected = False
                for button in self._mode_buttons.values():
                    button.state(["disabled"] if self._mode_selected else ["!disabled"])
                    if not self._mode_selected:
                        button.state(["!pressed"])
                self.start_button.state(
                    ["!disabled"]
                    if self._ready
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
        with self._graph_lock:
            values = self._pending_graph
            self._pending_graph = None
        if values is not None:
            self._update_realtime_graph(values)
        self.after(40, self._process_events)

    def _render_image(self, event=None):
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete("all")
        if self._image is None:
            self.canvas.create_text(
                width / 2, height / 2, text="In attesa di un'immagine", fill=self.colors["placeholder"]
            )
            return
        scale = min(width / self._image.width, height / self._image.height)
        size = (max(1, min(width, round(self._image.width * scale))),
                max(1, min(height, round(self._image.height * scale))))
        preview = self._image.resize(size, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(width / 2, height / 2, image=self._photo)


if __name__ == "__main__":
    LiralabGUI().mainloop()
