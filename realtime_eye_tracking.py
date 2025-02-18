import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import time

# Load pre-trained model
MODEL_PATH = 'models/model.h5'  # Replace with your trained model path
model = load_model(MODEL_PATH)

# Labels for classification
LABELS = ["Closed", "Forward", "Left", "Right"]

# Load OpenCV face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture
cap = cv2.VideoCapture(0)
log_file = open("student_status.log", "w", buffering=1)  # Line buffering for real-time updates
closed_eye_start_time = None
distracted_start_time = None
last_logged_status = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    status = "Active"

    if len(faces) == 0:
        if distracted_start_time is None:
            distracted_start_time = time.time()
        elif time.time() - distracted_start_time > 5:
            status = "Distracted"
    else:
        distracted_start_time = None  # Reset distraction timer

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            if closed_eye_start_time is None:
                closed_eye_start_time = time.time()
            elif time.time() - closed_eye_start_time > 5:
                status = "Not Awake"
            else:
                status = "Inactive (Sleeping)"
        else:
            closed_eye_start_time = None
            eye = roi_color[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
            eye = cv2.resize(eye, (80, 80))  # Resize to match model input
            eye = eye / 255.0  # Normalize
            eye = eye.reshape(1, 80, 80, 3)  # Reshape to (1, 80, 80, 3)
            
            prediction = model.predict(eye)
            state = LABELS[np.argmax(prediction)]
            closeCount =0 
            print(state)
            if state == "Closed":
                closeCount+=1
                if(closeCount>10):
                    status = "Not Awake"
                elif(closeCount>20):
                    status = "Inactive (Sleeping)"
            elif state != "Forward":
                    closeCount=0
                    status = "Distracted"
            else:
                closeCount = 0
                status = "Active"
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if status != last_logged_status:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {status}\n")
        last_logged_status = status
    
    cv2.imshow('Real-Time Eye Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
