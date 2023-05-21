import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
import os
import time
from plyer import notification

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_CLOSED_TIME_REQUIRED = 0.3
EYE_OPEN_EAR = 0.3

last_time_eyes_closed = None
alarm_playing = False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_dir = os.path.dirname(os.path.abspath(__file__))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat"))

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(os.path.join(current_dir, "alarm_sound.mp3"))

while True:
    ret, frame = cap.read()
    if frame is None:
        print("Failed to capture frame")
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    eyes_open = True

    if len(faces) > 0:
        face_detected_info = "Face Detected"
    else:
        face_detected_info = "No Face Detected"

    cv2.putText(frame, face_detected_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for (x, y, w, h) in faces:
        shape = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
        shape = face_utils.shape_to_np(shape)

        left_eye_ratio = eye_aspect_ratio(shape[42:48])
        right_eye_ratio = eye_aspect_ratio(shape[36:42])

        for i in range(36, 48):
            cv2.circle(frame, (shape[i][0], shape[i][1]), 2, (0, 255, 0), -1)

        EYE_AR_THRESH = 0.25

        if left_eye_ratio < EYE_AR_THRESH or right_eye_ratio < EYE_AR_THRESH:
            eyes_open = False

        left_eye_ratio_percentage = max(0, min(100, int((1.0 - left_eye_ratio / EYE_OPEN_EAR) * 100)))
        right_eye_ratio_percentage = max(0, min(100, int((1.0 - right_eye_ratio / EYE_OPEN_EAR) * 100)))

        cv2.putText(frame, f"Left Eye: {left_eye_ratio_percentage}%, Right Eye: {right_eye_ratio_percentage}%",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if not eyes_open:
        if last_time_eyes_closed is None:
            last_time_eyes_closed = time.time()
        elif time.time() - last_time_eyes_closed > EYE_CLOSED_TIME_REQUIRED:
            if not alarm_playing:
                alarm_sound.play()
                alarm_playing = True

                notification.notify(
                    title="Wake Up!",
                    message="Your eyes have been closed for too long!",
                    app_icon=None,
                    timeout=10,
                )
    else:
        last_time_eyes_closed = None
        if alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

    cv2.imshow("Wykrywanie Senności", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
