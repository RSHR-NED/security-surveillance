import os
import numpy as np
import cv2
import face_recognition as fr




# This class handles the encodings of the known faces in encdoingsDB.json file
class EncodingsDB:
    '''
    Handle the EncodingDB
    # Now, we consider one encoding of each face
    # Now, EncdoingsDB class only handle local self.encodings variable
    '''

    '''
    Required
    # handle the encodingDB.json file
    '''
    # this function loads the encodingsDF file
    def __init__(self) -> None:
        self.encodings = {}
    
    # find all the encodings of the given image
    # parameter -> images = { id: image_array, id : image array}
    def find_encodings(self, images:dict ): 
        for id in images:
            enc_id = fr.face_encodings(images[id])[0]
            self.encodings[ id ] = enc_id
    
    # add new encoding of the face to the EncodingDB
    def add_face_encoding(self, id:int, image:list):
        enc_id = fr.face_encodings(image)[0]
        self.encodings[ id ] = enc_id
    
    # delete encoding from the encodingDB
    def del_face_encoding(self, id):
        del self.encodings[ id ]
    
    # Now, work with each face with single encodings value
    def get_all_encodings(self):
        return np.array( list(self.encodings.values()) )
    
    # update the encodings of the given id 
    def set_face_encoding(self, id:int, image:list):
        pass




# This class handle the Images of the unidentifed images DB
class ImgDatabase:

    # read the images in the idenifiedDB and return dictionary of {id : image}
    # temporary
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



    # If the user marked the faces to be known, then it will use EncodingDB.add_encoding_face to add the faces in the identified faces of the Encdoings Databases
    def marked_known(self,id:int):
        pass
    


    # save image to the speified path
    def save_image(image,path):
        # upscaling the image by 4 b/c we downsize in detection
        resize_image  = cv2.resize(image, (200,200),interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path, resize_image)
    

