import csv
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

"""""""""
Usage:

> Read 'EPISODE' folder for .csv, /image and /mask and build hdf5 in 'HDF5_PATH' for ACT Training.

"""""""""
def sort_by_final_number(file_list):
    return sorted(file_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

EPISODES_PATH = "/home/legion/ROS/kinova_ws/AORTE"
episodes_list = [f for f in os.listdir(EPISODES_PATH) if os.path.isdir(os.path.join(EPISODES_PATH, f))]
e = 0
for EPISODE in sort_by_final_number(episodes_list):
    print("#"*50)
    print("#"*10 + f" PROCESSING EPISODE {EPISODE} " + "#"*10)
    print("#"*50)
    print(f"###### DONE {e}/{len(episodes_list)} episodes")

    MAIN_PATH = "/home/legion/ROS/kinova_ws/AORTE/" + EPISODE + "/"
    CSV_PATH = MAIN_PATH + EPISODE + ".csv"
    HDF5_PATH = MAIN_PATH + "../" + f"episode_{e}.hdf5"
    N = 600
    IMG_H = 256 # 480
    IMG_W = 256 # 640
    IMG_C = 3
    e += 1

    # Allocazione array
    actions = np.zeros((N, 6), dtype=np.float32)
    qpos = np.zeros((N, 9), dtype=np.float32)
    qvel = np.zeros((N, 7), dtype=np.float32)
    images = np.zeros((N, IMG_H, IMG_W, IMG_C), dtype=np.uint8)

    T_belly_0 = np.eye(4)   # Trasformazione da robot a ombelico (posizione iniziale)   [0T1]
    T_current_0 = np.eye(4) # Trasformazione da robot alla posizione attuale            [0T2]
    T_rotation_fix = np.eye(4) # Rotazione di 90° sull'asse Z per i vecchi csv (pre sensore di forza)

    with open(CSV_PATH, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if "zForce" not in reader.fieldnames:
            print(">> No force sensor here. SKIPPING")
            e -= 1
            continue
        #    T_rotation_fix = np.array([
        #    [0.0, 1.0, 0.0, 0.0],
        #    [-1.0,  0.0, 0.0, 0.0],
        #    [0.0,  0.0, 1.0, 0.0],
        #    [0.0,  0.0, 0.0, 1.0],
        #], dtype=float)
        

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
            
            force = np.array([
                float(row["xForce"]),
                float(row["yForce"]),
                float(row["zForce"])
            ])

            T_current_0[:3,:3] = Rmat
            T_current_0[0:3,3] = pos
            T_current_0 = T_current_0 @ T_rotation_fix

            if i == 0:
                T_belly_0[:3,:3] = Rmat
                T_belly_0[0:3,3] = pos

            T_current_belly = np.dot(np.linalg.inv(T_belly_0),T_current_0)
            
            euler = R.from_matrix(T_current_belly[:3,:3]).as_euler('xyz').astype(np.float32)
            ee_pose_belly = np.concatenate([T_current_belly[:3,3], euler])

            actions[i - 1 if i > 0 else i] = ee_pose_belly # PRIMA usavamo solo q
            qpos[i][:6] = ee_pose_belly
            qpos[i][6:] = force

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
        f.attrs["sim"] = True

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
print("\n\nFINISH\n\n")