import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('mask_model.h5')

# Set image size same as used in training
img_size = 224

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face - simple ROI in center (no face detector used here)
    h, w = frame.shape[:2]
    x1 = w // 2 - 112
    y1 = h // 2 - 112
    x2 = x1 + 224
    y2 = y1 + 224
    face = frame[y1:y2, x1:x2]

    # Preprocess the face
    face_input = cv2.resize(face, (img_size, img_size))
    face_input = face_input.astype('float32') / 255.0
    face_input = np.expand_dims(face_input, axis=0)

    # Predict
    pred = model.predict(face_input)[0][0]
    label = "With Mask" if pred < 0.5 else "Without Mask"
    color = (0, 255, 0) if pred < 0.5 else (0, 0, 255)

    # Draw rectangle and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
