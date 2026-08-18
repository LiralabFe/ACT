
import pandas as pd
import matplotlib.pyplot as plt

# Leggi il CSV
df = pd.read_csv("/home/legion/ROS/kinova_ws/AAA_SF_30/AAA_SF_30.csv")

# Converti il tempo da nanosecondi a secondi
t = (df["timestamp"] - df["timestamp"].iloc[0]) * 1e-9

# Plot
plt.figure(figsize=(10, 6))
print(df["zForce"][0])
plt.plot(df["zForce"], label="Fz")
plt.show()

"""
import cv2
import numpy as np
import torch
from liralab.utils.segmentator import Segmentator

cap = cv2.VideoCapture(0)

segmentator = Segmentator(
    "segmentation_models/unetplusplus/unetplusplus_imagenet_jugular.pth",
    "UnetPP"
)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    with torch.no_grad():
        mask = segmentator.get_segmented_mask(frame)

    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().detach().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255

    # Porta il frame alla dimensione della mask
    frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]))

    # Overlay rosso
    overlay = frame.copy()
    overlay[mask > 0] = (0, 0, 255)

    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    result = cv2.resize(result, (500, 500))
    cv2.imshow("Overlay", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

"""
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

# ======================
# Configurazione
# ======================
H5_PATH = "/home/legion/ROS/kinova_ws/AORTE/episode_1.hdf5"
AXIS_LENGTH = 0.03

# ======================
# Lettura HDF5
# ======================
with h5py.File(H5_PATH, "r") as f:
    qpos = f["observations/qpos"][:]

positions = qpos[:, :3]
rpy = qpos[:, 3:]   # roll, pitch, yaw

# Rotazioni (XYZ = roll, pitch, yaw)
rotations = R.from_euler("xyz", rpy).as_matrix()

# ======================
# Figura
# ======================
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

mins = positions.min(axis=0)
maxs = positions.max(axis=0)

margin = 0.05
ax.set_xlim(mins[0]-margin, maxs[0]+margin)
ax.set_ylim(mins[1]-margin, maxs[1]+margin)
ax.set_zlim(mins[2]-margin, maxs[2]+margin)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Stessa scala sui tre assi
center = (mins + maxs) / 2
radius = np.max(maxs - mins) / 2 + margin

ax.set_xlim(center[0]-radius, center[0]+radius)
ax.set_ylim(center[1]-radius, center[1]+radius)
ax.set_zlim(center[2]-radius, center[2]+radius)

# Traiettoria
ax.plot(
    positions[:,0],
    positions[:,1],
    positions[:,2],
    color="gray",
    alpha=0.3
)

point, = ax.plot([], [], [], "ko", markersize=6)

x_axis, = ax.plot([], [], [], "r-", lw=3)
y_axis, = ax.plot([], [], [], "g-", lw=3)
z_axis, = ax.plot([], [], [], "b-", lw=3)

def update(i):

    p = positions[i]
    Rmat = rotations[i]

    point.set_data([p[0]], [p[1]])
    point.set_3d_properties([p[2]])

    ex = p + AXIS_LENGTH * Rmat[:,0]
    ey = p + AXIS_LENGTH * Rmat[:,1]
    ez = p + AXIS_LENGTH * Rmat[:,2]

    x_axis.set_data([p[0], ex[0]], [p[1], ex[1]])
    x_axis.set_3d_properties([p[2], ex[2]])

    y_axis.set_data([p[0], ey[0]], [p[1], ey[1]])
    y_axis.set_3d_properties([p[2], ey[2]])

    z_axis.set_data([p[0], ez[0]], [p[1], ez[1]])
    z_axis.set_3d_properties([p[2], ez[2]])

    ax.set_title(f"Frame {i}")

    return point, x_axis, y_axis, z_axis

ani = FuncAnimation(
    fig,
    update,
    frames=len(positions),
    interval=30,
    blit=False
)

plt.show()
"""
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================
# Configurazione
# ======================
CSV_PATH = "/home/legion/ROS/kinova_ws/OLD_AORTE/AAA_AP_3/AAA_AP_3.csv"   # <-- cambia con il tuo file
AXIS_LENGTH = 0.03            # lunghezza degli assi disegnati

# ======================
# Lettura CSV
# ======================
df = pd.read_csv(CSV_PATH)

positions = df[["x", "y", "z"]].to_numpy()

rotations = df[[
    "r11", "r12", "r13",
    "r21", "r22", "r23",
    "r31", "r32", "r33"
]].to_numpy().reshape(-1, 3, 3)

# ======================
# Figura
# ======================
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Limiti automatici
mins = positions.min(axis=0)
maxs = positions.max(axis=0)

margin = 0.05
ax.set_xlim(mins[0]-margin, maxs[0]+margin)
ax.set_ylim(mins[1]-margin, maxs[1]+margin)
ax.set_zlim(mins[2]-margin, maxs[2]+margin)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Traiettoria completa
ax.plot(
    positions[:,0],
    positions[:,1],
    positions[:,2],
    color="gray",
    alpha=0.3
)

# Punto corrente
point, = ax.plot([], [], [], 'ko')

# Assi locali
x_axis, = ax.plot([], [], [], 'r-', lw=3)
y_axis, = ax.plot([], [], [], 'g-', lw=3)
z_axis, = ax.plot([], [], [], 'b-', lw=3)


def update(i):

    p = positions[i]
    R = rotations[i]

    point.set_data([p[0]], [p[1]])
    point.set_3d_properties([p[2]])

    ex = p + AXIS_LENGTH * R[:,0]
    ey = p + AXIS_LENGTH * R[:,1]
    ez = p + AXIS_LENGTH * R[:,2]

    x_axis.set_data([p[0], ex[0]], [p[1], ex[1]])
    x_axis.set_3d_properties([p[2], ex[2]])

    y_axis.set_data([p[0], ey[0]], [p[1], ey[1]])
    y_axis.set_3d_properties([p[2], ey[2]])

    z_axis.set_data([p[0], ez[0]], [p[1], ez[1]])
    z_axis.set_3d_properties([p[2], ez[2]])

    ax.set_title(f"Frame {i}")
    print(rotations[i])
    print("----")
    return point, x_axis, y_axis, z_axis


ani = FuncAnimation(
    fig,
    update,
    frames=len(df),
    interval=30,
    blit=False
)

plt.show()
"""
"""
import h5py
import matplotlib.pyplot as plt

# Percorso del file
h5_file = "/home/legion/ROS/kinova_ws/AORTE/AAA_APR_1/episode_5.hdf5"

# Leggi il dataset
with h5py.File(h5_file, "r") as f:
    qpos = f["observations/qpos"][:]

# Estrai la coordinata z
z = qpos[:, 2]
y = qpos[:, 1]
x = qpos[:, 0]

# Plot
plt.figure(figsize=(12, 5))
plt.plot(x, linewidth=1.5)
plt.plot(y, linewidth=1.5)
plt.plot(z, linewidth=1.5)

plt.title("Posizione z nel tempo")
plt.xlabel("Frame")
plt.ylabel("z")
plt.grid(True)
plt.tight_layout()

plt.show()
"""
"""
import pandas as pd
import matplotlib.pyplot as plt

# Percorso del file CSV
csv_file = "/home/legion/ROS/kinova_ws/AORTE/AAA_JR_1/AAA_JR_1.csv"

# Leggi il CSV
df = pd.read_csv(csv_file)

# Se il timestamp è in formato numerico lascia così,
# altrimenti prova a convertirlo in datetime.
try:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
except Exception:
    pass

# Plot di z nel tempo
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["x"], linewidth=1.5)
plt.plot(df["timestamp"], df["y"], linewidth=1.5)
plt.plot(df["timestamp"], df["z"], linewidth=1.5)

plt.title("Andamento di z nel tempo")
plt.xlabel("Timestamp")
plt.ylabel("z")
plt.grid(True)
plt.tight_layout()

plt.show()
"""