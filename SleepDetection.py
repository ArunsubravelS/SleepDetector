import cv2
import tkinter as tk
from tkinter import messagebox
import time
import pygame

def ask_permission():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    result = messagebox.askyesno("Permission Request", "Do you allow access to the camera?")
    root.destroy()
    return result

def play_alert_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("E:/python/SleepDetector/alert.mp3")
    pygame.mixer.music.play()

if ask_permission():
    # Load the pre-trained Haar Cascade classifiers for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        # Define the codec and create a VideoWriter object to save the video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

        last_blink_time = time.time()
        blink_threshold = 5  # seconds to trigger the alert if no blink is detected

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                break

            # Convert the frame to grayscale (face and eye detection works better on grayscale images)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            eyes_detected = False

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Region of interest for eyes within the face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # Detect eyes within the face region
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) > 0:
                    eyes_detected = True

            if eyes_detected:
                last_blink_time = time.time()  # reset the timer on eye detection

            if time.time() - last_blink_time > blink_threshold:
                play_alert_sound()
                last_blink_time = time.time()  # reset the timer after alert

            # Display the resulting frame
            cv2.imshow('Face and Blink Detection', frame)

            # Write the frame to the output video file
            out.write(frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and the video writer
        cap.release()
        out.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()
else:
    print("Camera access denied by user.")
