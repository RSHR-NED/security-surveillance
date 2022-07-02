import cv2
import face_recognition as fr
from .handle_databases import ImgDatabase


class FaceDetection:

    def __init__(self) -> None:
        self.RGB_resized_frame = None
        self.faces_loc = None

    def detect_faces(self, frame, resize = 0.25 ):
        # down sampling of an image b/c of optimizing time in which INTER_AREA is done great in downsizing
        resized_frame = cv2.resize(frame, (0,0), None, resize , resize, interpolation=cv2.INTER_AREA)
        self.RGB_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB )
        self.faces_loc = fr.face_locations(self.RGB_resized_frame)
        if self.faces_loc:
            return self.faces_loc
        else:
            return None
    
    # encode the faces of the image given the location of the faces
    def encode_detect_faces(self):
        encoded = fr.face_encodings(self.RGB_resized_frame, self.faces_loc)
        return encoded
    

class FaceRecognition:

    def __init__(self) -> None:
        self.FD = FaceDetection()
        ImgDatabase()

    
    # compares the each faces of current frame with the identified databases
    def comparing_faces(self, frame, encodingsDB):
        
        detected_faces = self.FD.detect_faces(frame)
        if detected_faces:
            encoded_faces = self.FD.encode_detect_faces()
            match  = fr.compare_faces(encodingsDB, encoded_faces)
            print(match)
            # still more to code
        else:
            return "Face Not Detected"
