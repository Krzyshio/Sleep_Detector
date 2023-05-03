import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
import os

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        shape = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
        shape = face_utils.shape_to_np(shape)

        left_eye_ratio = eye_aspect_ratio(shape[42:48])
        right_eye_ratio = eye_aspect_ratio(shape[36:42])

        for i in range(36, 48):
            cv2.circle(frame, (shape[i][0], shape[i][1]), 2, (0, 255, 0), -1)

        EYE_AR_THRESH = 0.25

        if left_eye_ratio < EYE_AR_THRESH or right_eye_ratio < EYE_AR_THRESH:
            alarm_sound.play()

    cv2.imshow("Wykrywanie Senności", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()