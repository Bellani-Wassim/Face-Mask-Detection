import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mask_model.h5')
img_size = 224

#Load Haar Cascade Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Start caputre
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # convert the frame to gray because haar cascade expect a gray frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # crop the face
        face = frame[y:y+h, x:x+w]
        # resize
        face_input = cv2.resize(face, (img_size, img_size))
        # normalise 
        face_input = face_input.astype('float32') / 255.0
        # adds a batch dimension
        face_input = np.expand_dims(face_input, axis=0)

        pred = model.predict(face_input)[0][0]
        label = "With mask" if pred < 0.5 else "Without mask"
        color = (0, 255, 0) if pred < 0.5 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()