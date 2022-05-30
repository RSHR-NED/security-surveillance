from time import time
import os
import cv2
import face_recognition as fr
import numpy as np


class Encodings:

    def __init__(self) -> None:
        self.encodings = {}
    
    def find_encodings(self, images:dict ):
        for id in images:
            enc_id = fr.face_encodings(images[id])[0]
            self.encodings[ id ] = enc_id
    
    def add_face_encoding(self, id, image):
        enc_id = fr.face_encodings(image)[0]
        self.encodings[ id ] = enc_id
    
    def del_face_encoding(self, id):
        del self.encodings[ id ]
    
    # Now, work with each face with single encodings value
    def get_all_encodings(self):
        return self.encodings.values()
    
    def set_face_encoding(self, id):
        pass





class ImgDatabase:


    def read_identified_faces_db(path):
        # each id with images,arr
        identified_faces = {}

        for img in os.listdir(path):
            # each image loading
            img_arr = fr.load_image_file( "./identified_faces/" +img )
            # It load the image in BGR format So, 
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            # scrap image id
            img_id = img.split(".")[0]
            # added to the dictionary
            identified_faces[ img_id ] = img_arr
        
        return identified_faces
    

    def read_unidentified_faces_db(path):

        unidentified_faces = {}

        for img in os.listdir(path):
            # each image loading
            img_arr = fr.load_image_file( "./unidentified_faces/" +img )
            # It load the image in BGR format So, 
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            # scrap image id
            img_id = img.split(".")[0]
            # added to the dictionary
            unidentified_faces[ img_id ] = img_arr
        
        return unidentified_faces

    def marked_known(self,id):
        
        pass

class Detection:

    def __init__(self) -> None:
        self.faces = None

    def detect_faces(self, frame):
        fr.face_locations(frame, 0.25,)
    







images = ImgDatabase.read_identified_faces_db("./identified_faces")
E = Encodings()
E.find_encodings(images)
encodings = np.array( list(E.get_all_encodings()) )


image_M_test = fr.load_image_file('unidentified_faces/Slushii_test.png')
image_M_test = cv2.cvtColor(image_M_test, cv2.COLOR_BGR2RGB)
encoded= fr.face_encodings(image_M_test)[0]
start = time() 
print(fr.compare_faces(encodings, encoded))
end = time()
print("Execution Time:", end-start)



    


            


