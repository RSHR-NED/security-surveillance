import os
import numpy as np
import cv2
import face_recognition as fr
import json


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

    def __init__(self) -> None:
        # load ecodingDB.json file
        with open('encodingsDB.json') as json_file:
            self.encodings = json.load(json_file)

    def find_encodings(self, images: dict):
        ''' find all the encodings of the given image
        parameter -> images = { id: image_array, id : image array}
        '''
        for id_ in images:
            enc_id = fr.face_encodings(images[id_])[0]
            self.encodings[id_] = enc_id

    def save_encodings(self):

        # convert each encoding from numpy array to list
        for id_ in self.encodings:
            self.encodings[id_] = self.encodings[id_].tolist()
            
        # save the encodings to the encodingDB.json file
        with open('encodingsDB.json', 'w') as outfile:
            json.dump(self.encodings, outfile)

    def add_face_encoding(self, id_: int, image: list):
        '''
        adds new encoding of a face to the EncodingDB
         '''
        enc_id = fr.face_encodings(image)[0]
        self.encodings[id_] = enc_id
        self.save_encodings()

    def del_face_encoding(self, id_):
        '''
        Given an id, deletes the corresponding encoding
        '''
        del self.encodings[id_]
        self.save_encodings()

    # Now, work with each face with single encodings value
    def get_all_encodings(self):
        return np.array(list(self.encodings.values()))

    # update the encodings of the given id
    def set_face_encoding(self, id_: int, image: list):
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
            img_arr = fr.load_image_file(("identified_faces/" + img))
            # It load the image in BGR format So,
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            # scrap image id
            img_id = img.split(".")[0]
            # added to the dictionary
            identified_faces[img_id] = img_arr

        return identified_faces

    # read the images in the unidenifiedDB and return dictionary of {id : image}

    def read_unidentified_faces_db(path):

        unidentified_faces = {}

        for img in os.listdir(path):
            # each image loading
            img_arr = fr.load_image_file("./unidentified_faces/" + img)
            # It load the image in BGR format So,
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            # scrap image id
            img_id = img.split(".")[0]
            # added to the dictionary
            unidentified_faces[img_id] = img_arr

        return unidentified_faces

    # If the user marked the faces to be known, then it will use EncodingDB.add_encoding_face to add the faces in the identified faces of the Encdoings Databases

    def marked_known(self, id_: int):
        pass

    # save image to the speified path

    def save_image(image, path):
        # upscaling the image by 4 b/c we downsize in detection
        resize_image = cv2.resize(
            image, (200, 200), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path, resize_image)
