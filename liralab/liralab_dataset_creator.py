import csv
import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

EPISODE = "AAA_SF_1"
MAIN_PATH = "/home/legion/ROS/kinova_ws/AORTE/" + EPISODE + "/"
CSV_PATH = MAIN_PATH + EPISODE + ".csv"
HDF5_PATH = MAIN_PATH + EPISODE + ".hdf5"
SIM_VALUE = True  # attributo 'sim'

N = 600
IMG_H = 480
IMG_W = 640
IMG_C = 3

# Allocazione array
actions = np.zeros((N, 6), dtype=np.float32)
qpos = np.zeros((N, 6), dtype=np.float32)
qvel = np.zeros((N, 7), dtype=np.float32)
images = np.zeros((N, IMG_H, IMG_W, IMG_C), dtype=np.uint8)

T_belly_0 = np.eye(4)   # Trasformazione da robot a ombelico (posizione iniziale)   [0T1]
T_current_0 = np.eye(4) # Trasformazione da robot alla posizione attuale            [0T2]

with open(CSV_PATH, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    
    for i, row in enumerate(reader):
        if i >= N:
            break

        # qpos e action = [q0 ... q6]
        q = np.array([float(row[f"q{j}"]) for j in range(7)], dtype=np.float32)
        pos = np.array([
            float(row["x"]),
            float(row["y"]),
            float(row["z"])
        ], dtype=np.float32)

        Rmat = np.array([
            [float(row["r11"]), float(row["r12"]), float(row["r13"])],
            [float(row["r21"]), float(row["r22"]), float(row["r23"])],
            [float(row["r31"]), float(row["r32"]), float(row["r33"])]
        ])

        T_current_0[:3,:3] = Rmat
        T_current_0[0:3,3] = pos

        if i == 0:
            T_belly_0[:3,:3] = Rmat
            T_belly_0[0:3,3] = pos

        T_current_belly = np.dot(np.linalg.inv(T_belly_0),T_current_0)
        
        euler = R.from_matrix(T_current_belly[:3,:3]).as_euler('xyz').astype(np.float32)
        ee_pose_belly = np.concatenate([T_current_belly[:3,3], euler])

        actions[i - 1 if i > 0 else i] = ee_pose_belly # PRIMA usavamo solo q
        qpos[i] = ee_pose_belly

        # Caricamento immagine
        img_path = MAIN_PATH + "/image/" + row["image"]
        mask_path = MAIN_PATH + "/mask/mask_" + row["image"]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        img = img.resize((IMG_W, IMG_H))
        mask = mask.resize((IMG_W, IMG_H))
        
        # Converti in numpy array
        r = np.array(img, dtype=np.uint8)
        g = np.array(mask, dtype=np.uint8)
        b = np.zeros_like(r, dtype=np.uint8)

        # Stack nei canali RGB
        rgb = np.stack([r, g, b], axis=2)

        images[i] = np.array(rgb, dtype=np.uint8)

# Scrittura HDF5
with h5py.File(HDF5_PATH, "w") as f:
    # attributi
    f.attrs["sim"] = SIM_VALUE

    # dataset
    f.create_dataset(
        "action",
        data=actions,
        dtype="float32",
    )

    f.create_dataset(
        "observations/qpos",
        data=qpos,
        dtype="float32",
    )

    f.create_dataset(
        "observations/qvel",
        data=qvel,
        dtype="float32",
    )

    f.create_dataset(
        "observations/images/top",
        data=images,
        dtype="uint8",
    )

print("HDF5 creato con successo:", HDF5_PATH)
