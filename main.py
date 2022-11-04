from helpers import FaceRecognizer
from time import time
import cv2

# for mobile camera feed
import numpy as np  
import urllib.request


def main() -> None:
    face_recognizer = FaceRecognizer()
    URL = "https://192.168.18.56:8080/video"  # mobile camera feed url, from ip webcam app
    use_mobile_camera = False
    if use_mobile_camera:
        cam = cv2.VideoCapture(URL)  # mobile camera feed
    else:
        cam = cv2.VideoCapture(0)  # laptop web cam feed

    while True:

        # to exit by pressing "q" or esc
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
            break

        ret, frame = cam.read()  # read a frame
        if not ret:  # if frame is not read correctly
            print("Error: Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)  # mirror frame horizontally
        if use_mobile_camera:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame

        # recognize faces in the frame
        frame_faces_labels, frame_face_locations = face_recognizer.recognize_faces(frame)

        # draw rectangles around faces
        for (top, right, bottom, left), label in zip(frame_face_locations, frame_faces_labels):
            # draw box and label around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)  
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow('Face Recognizer', frame)  # display the resulting frame


        
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()