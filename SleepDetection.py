import cv2
import tkinter as tk
from tkinter import messagebox
import time
import pygame
import numpy as np
import mediapipe as mp
from scipy.spatial import distance

def ask_permission():
    root = tk.Tk()
    root.withdraw()  # <---Hide the root window
    result = messagebox.askyesno("Permission Request", "Do you allow access to the camera?")
    root.destroy()
    return result

def play_alert_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("E:/python/SleepDetector/Sound/alert.mp3")
    pygame.mixer.music.play()

def stop_alert_sound():
    pygame.mixer.music.stop()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

if ask_permission():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

        last_blink_time = time.time()
        blink_threshold = 5
        ear_threshold = 0.25
        ear_consec_frames = 3
        counter = 0
        total_frames = 0
        drowsy_frames = 0
        alert_playing = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            total_frames += 1

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    left_eye = [(landmarks[i].x, landmarks[i].y) for i in [362, 385, 387, 263, 373, 380]]
                    right_eye = [(landmarks[i].x, landmarks[i].y) for i in [33, 160, 158, 133, 153, 144]]

                    left_eye = np.array([(int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])) for point in left_eye])
                    right_eye = np.array([(int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])) for point in right_eye])

                    leftEAR = eye_aspect_ratio(left_eye)
                    rightEAR = eye_aspect_ratio(right_eye)
                    ear = (leftEAR + rightEAR) / 2.0

                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if ear < ear_threshold:
                        counter += 1
                        drowsy_frames += 1
                        if counter >= ear_consec_frames:
                            if not alert_playing:
                                play_alert_sound()
                                alert_playing = True
                    else:
                        counter = 0
                        if alert_playing:
                            stop_alert_sound()
                            alert_playing = False

                    cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=1)
                    cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=1)

            drowsiness_percentage = (drowsy_frames / total_frames) * 100
            cv2.putText(frame, "Drowsiness: {:.2f}%".format(drowsiness_percentage), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Drowsiness Detection', frame)
            out.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                stop_alert_sound()
                alert_playing = False

        cap.release()
        out.release()
        cv2.destroyAllWindows()
else:
    print("Camera access denied by user.")
