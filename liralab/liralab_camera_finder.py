import cv2

# Apre la webcam (0 = webcam di default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Errore: impossibile aprire la webcam")
    exit()

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