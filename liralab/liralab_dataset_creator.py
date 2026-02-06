import csv
import h5py
import numpy as np
from PIL import Image

EPISODE = "episode_5"
MAIN_PATH = "/home/legion/ROS/kinova_ws/" + EPISODE
CSV_PATH = MAIN_PATH + "/test.csv"
HDF5_PATH = MAIN_PATH + "/" + EPISODE + ".hdf5"
SIM_VALUE = True  # attributo 'sim'

N = 400
IMG_H = 480
IMG_W = 640
IMG_C = 3

# Allocazione array
actions = np.zeros((N, 7), dtype=np.float32)
qpos = np.zeros((N, 7), dtype=np.float32)
qvel = np.zeros((N, 7), dtype=np.float32)
images = np.zeros((N, IMG_H, IMG_W, IMG_C), dtype=np.uint8)

with open(CSV_PATH, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    
    for i, row in enumerate(reader):
        if i >= N:
            break

        # qpos e action = [q0 ... q6]
        q = np.array([
            float(row[f"q{j}"]) for j in range(7)
        ], dtype=np.float32)

        actions[i - 1 if i > 0 else i] = q
        qpos[i] = q

        # Caricamento immagine
        img_path = MAIN_PATH + "/image/" + row["image"]
        print(img_path)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_W, IMG_H))  # sicurezza
        images[i] = np.array(img, dtype=np.uint8)

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
