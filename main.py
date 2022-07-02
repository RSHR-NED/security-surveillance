import imp
from detect_n_recognize.face_dnr import *
from detect_n_recognize.handle_databases import*
from time import time

def main() -> None:    
    E = EncodingsDB()
    FR = FaceRecognition()
    cam = cv2.VideoCapture(0)
    FRAME_RATE = 7
    prev = 0
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

        images = ImgDatabase.read_identified_faces_db("./identified_faces")
        E.find_encodings(images)
        encodings = E.get_all_encodings()

        FR.comparing_faces(frame,encodings)
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