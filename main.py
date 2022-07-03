from helpers import FaceRecognizer
from time import time
import cv2

def main() -> None:
    face_recognizer = FaceRecognizer()
    cam = cv2.VideoCapture(0)
    FRAME_RATE = 30
    prev = 0
    while True:

        # to exit by pressing "q" or esc
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
            break
        
        # limiting to set frame rate
        time_elapsed = time() - prev
        if time_elapsed <= 1/FRAME_RATE:
            continue  # skip this iteration

        prev = time()

        ret, frame = cam.read()  # read a frame
        if not ret:  # if frame is not read correctly
            print("Error: Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)  # mirror frame horizontally

        # recognize faces in the frame
        frame_faces_labels, frame_face_locations = face_recognizer.recognize_faces(frame)

        # draw rectangles around faces
        for (top, right, bottom, left), name in zip(frame_face_locations, frame_faces_labels):
            # # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            # top *= 4
            # right *= 4
            # bottom *= 4
            # left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # display the resulting frame
        cv2.imshow('Face Recognizer', frame)


        
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()