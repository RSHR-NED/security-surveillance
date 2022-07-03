from helpers import FaceRecognizer
from time import time
import cv2

def main() -> None:
    face_recognizer = FaceRecognizer()
    cam = cv2.VideoCapture(0)
    FRAME_RATE = 30
    process_frame = True
    prev = 0
    while True:

        # to exit by pressing "q" or esc
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
            break
        
        # # limiting to set frame rate
        # time_elapsed = time() - prev
        # if time_elapsed <= 1/FRAME_RATE:
        #     continue  # skip this iteration

        # prev = time()
        process_frame = not process_frame
        if not process_frame:
            print("Not processing")
            continue
        print("Processing")
        ret, frame = cam.read()  # read a frame
        if not ret:  # if frame is not read correctly
            print("Error: Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)  # mirror frame horizontally

        # recognize faces in the frame
        frame_faces_labels, frame_face_locations = face_recognizer.recognize_faces(frame)

        # draw rectangles around faces
        for (top, right, bottom, left), name in zip(frame_face_locations, frame_faces_labels):
            # draw box and label around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)  
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow('Face Recognizer', frame)  # display the resulting frame


        
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()