
from time import time
import os
import cv2
import face_recognition as fr
import numpy as np


class EncodingsDB:
    '''
    Handle the EncodingDB
    # Now, we consider one encoding of each face
    '''
    def __init__(self) -> None:
        self.encodings = {}
    
    # find all the encodings of the given image dictionary
    def find_encodings(self, images:dict ):
        for id in images:
            enc_id = fr.face_encodings(images[id])[0]
            self.encodings[ id ] = enc_id
    
    # add new encoding to the EncodingDB
    def add_face_encoding(self, id, image):
        enc_id = fr.face_encodings(image)[0]
        self.encodings[ id ] = enc_id
    
    # delete encoding from the encodingDB
    def del_face_encoding(self, id):
        del self.encodings[ id ]
    
    # Now, work with each face with single encodings value
    def get_all_encodings(self):
        return np.array( list(self.encodings.values()) )
    
    # update the encodings
    def set_face_encoding(self, id):
        pass





class ImgDatabase:

    # read the images in the idenifiedDB and return dictionary of {id : image}
    def read_identified_faces_db(path):
        # each id with images,arr
        identified_faces = {}

        for img in os.listdir(path):
            # each image loading
            img_arr = fr.load_image_file( ("identified_faces/" + img) )
            # It load the image in BGR format So, 
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            # scrap image id
            img_id = img.split(".")[0]
            # added to the dictionary
            identified_faces[ img_id ] = img_arr
        
        return identified_faces
    

    # read the images in the unidenifiedDB and return dictionary of {id : image}
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



    # If the user marked the faces to be known, then it move the face from identifiedDB and add encoding of that face in the EncodingDB
    def marked_known(self,id):
        pass
    


    # save image to the speified path
    def save_image(image,path):
        # upscaling the image by 4 b/c we downsize in detection
        resize_image  = cv2.resize(image, (100,100),interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path, resize_image)
    


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
        self.ImgDB = ImgDatabase()

    
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







    
    

    
            



    





# Constant

images = ImgDatabase.read_identified_faces_db("./identified_faces")
E = EncodingsDB()
E.find_encodings(images)
encodings = E.get_all_encodings()


# Without Modular Approach

""" image_M_test = fr.load_image_file('unidentified_faces/Valentin_test.jpg')
start = time()
# image_M_test = cv2.resize(image_M_test, (0,0), None, 0.25 , 0.25, interpolation=cv2.INTER_AREA)
image_M_test = cv2.cvtColor(image_M_test, cv2.COLOR_BGR2RGB)
faces = fr.face_locations(image_M_test)[0]
print(faces)
encoded= fr.face_encodings(image_M_test, [faces])[0]
#print(encoded)
print(fr.compare_faces(encodings, encoded))
end = time()
print("Execution Time:", end-start) """



""" # With Modular Approach    

image_M_test = fr.load_image_file('unidentified_faces/Valentin_test.jpg')

start = time() 
FR = FaceRecognition()
FR.comparing_faces(image_M_test,encodings)
end = time()
print("Execution Time:", end-start) """






 