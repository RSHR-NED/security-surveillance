import cv2
from time import time
from deepface import DeepFace
import numpy as np


def main() -> None:
    knowledge = get_knowledge()
    
    # DeepFace.stream(db_path = "./unidentified_faces")

    cam = cv2.VideoCapture(0)
    FRAME_RATE = 7
    prev = 0
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:

        # to exit by pressing "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # limiting to set frame rate
        time_elapsed = time() - prev
        if time_elapsed <= 1/FRAME_RATE:
            continue  # skip this iteration
        
        prev = time()

        # read a frame
        ret, frame = cam.read()
        if not ret:  # if frame is not read correctly
            print("Error: Failed to read frame from camera")
            break
        frame = cv2.flip(frame, 1)  # mirror frame horizontally

        # run face detection algorithm
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale for face detection
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)  # Detect faces

        # for every face detected
        for face in faces:
            x, y, w, h = face  # get face coordinates
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # draw rectangle around face

            # face = frame[y:y+h, x:x+w]  # extract face from frame
            
            # face = np.array(face)  # convert to numpy array (for deepface)
            # recognition = DeepFace.find(face, db_path = "./unidentified_faces")  # recognize face
            # print(recognition)
   
        cv2.imshow("Face Recognition", frame)  # display the current frame
        
    cam.release()
    cv2.destroyAllWindows()



def get_knowledge() -> dict:
    """
    Returns a dictionary of all the knowledge.json file
    """
    pass


if __name__ == "__main__":
    main()

