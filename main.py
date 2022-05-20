import cv2
from time import time

def main() -> None:
    knowledge = get_knowledge()

    # live camera feed loop
    # Loop indefinetely
        # if exit key or cross button is pressed, break loop
        # show live camera feed
        # run face detection algorithm
        # if atleast 1 face detected:
            # for each face:
                # run face recognition algorithm against identified_faces database
                # if face is recognized / present in identified_faces database
                    # get known/unknown label from knowledge dictionary
                    # mark face on frame with known/unknown label
                # if face is not recognized / not present in identified_faces database
                    # mark face on frame with unknown label
                    # run face recognition algorithm on each face in unidentified_faces database
                    # if the face is not in unidentified_faces database:
                        # add face to unidentified_faces database

    # show webcam
    cam = cv2.VideoCapture(0)
    FRAME_RATE = 10
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

        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break
        frame = cv2.flip(frame, 1)

        # run face detection algorithm
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
        cv2.imshow("Face Recognition", frame)
        prev = time()
        
    cam.release()
    cv2.destroyAllWindows()




def get_knowledge() -> dict:
    """
    Returns a dictionary of all the knowledge.json file
    """
    pass


if __name__ == "__main__":
    main()

