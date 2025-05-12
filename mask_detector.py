import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mask_model.h5')

img_size = 224

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    h, w = frame.shape[:2]

    # Define a 224x224 square in the center
    x1 = w // 2 - 112
    y1 = h // 2 - 112
    x2 = x1 + 224
    y2 = y1 + 224

    # Extract the face region (ROI)
    face = frame[y1:y2, x1:x2]

    # Preprocess the face
    face_input = cv2.resize(face, (img_size, img_size))
    face_input = face_input.astype('float32') / 255.0
    face_input = np.expand_dims(face_input, axis=0)

    # Predict using the model
    pred = model.predict(face_input)[0][0]
    label = "With Mask" if pred < 0.5 else "Without Mask"
    color = (0, 255, 0) if pred < 0.5 else (0, 0, 255)

    # Draw guide rectangle (yellow) with instruction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(frame, "Put your face here", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show prediction result below the box
    cv2.putText(frame, label, (x1, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the webcam window
    cv2.imshow("Mask Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close window
cap.release()
cv2.destroyAllWindows()