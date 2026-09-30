import cv2
import sys

# Controlla che sia stato specificato l'indice della webcam
if len(sys.argv) < 2:
    print("Uso: python webcam.py <indice_camera>")
    print("Esempio: python webcam.py 4")
    sys.exit(1)

# Legge l'indice della webcam dal primo argomento
camera_index = int(sys.argv[1])

# Apre la webcam
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Errore: impossibile aprire la webcam {camera_index}")
    sys.exit(1)

while True:
    # Legge un frame dalla webcam
    ret, frame = cap.read()

    if not ret:
        print("Errore nella lettura del frame")
        break

    # Mostra il frame in una finestra
    cv2.imshow("Webcam", frame)

    # Esce premendo il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la webcam e chiude le finestre
cap.release()
cv2.destroyAllWindows()