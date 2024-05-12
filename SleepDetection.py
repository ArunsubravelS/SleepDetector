

import cv2

def capture_and_process_frames():
    # Open the webcam
    cap = cv2.VideoCapture(0)  # Change the parameter to 0 for the default webcam or 1 for an external webcam
    
    if not cap.isOpened():
        print("Error: Webcam not opened correctly.")
        return


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Preprocess frame (optional)
        # For example, you might resize the frame to match the input size expected by your model

        # Perform inference with your machine learning model
        # For demonstration purposes, let's just display the original frame
        cv2.imshow(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

capture_and_process_frames()