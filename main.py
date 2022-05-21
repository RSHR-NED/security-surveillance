import cv2
from time import time

def main() -> None:
    knowledge = get_knowledge()

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
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
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

