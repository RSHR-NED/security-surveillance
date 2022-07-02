import json
import os
from PIL import UnidentifiedImageError

import cv2
import face_recognition
import numpy as np


class FaceRecognizer:
    def __init__(self):
        """
        Initializes the FaceRecognizer.
        """
        print("Initializing FaceRecognizer...\n\n")
        self.identified_faces, self.unidentified_faces = self.load_face_encodings()
        
        self.newest_id = 0 if len(self.identified_faces) == 0 else max(self.unidentified_faces.keys())
        self.newest_unid = 0 if len(self.unidentified_faces) == 0 else max(self.unidentified_faces.keys())
        print("FaceRecognizer initialized.\n\n")
        

    def load_face_encodings(self):
        """
        Loads the face encodings for identified and unidentified faces.
        For both, identified and unidentified faces, it first tries to load from json file.
        If that fails, it tries to load from the respective image folder.
        """

        # load unidentified faces
        print("Loading unidentified faces...")
        unidentified_faces = {}  # {name: image encoding [list]}
        with open("unidentified_faces_encodings.json", "r") as f:
            try:
                unidentified_faces = json.load(f)
            except json.decoder.JSONDecodeError:
                print("Failed to load unidentified faces from json file.")
                print("Loading from image folder...")
                for filename in os.listdir("./unidentified_faces/"):  # for each image in the folder
                    image_array = face_recognition.load_image_file(("./unidentified_faces/" + filename))  # load image
                    image_encoding = face_recognition.face_encodings(image_array)[0].tolist()  # encode the image, assuming only one face in image
                    id_ = filename.split(".")[0]  # get the name of the image (filename without extension)
                    unidentified_faces[id_] = image_encoding  # add the image encoding to the dictionary
                self.save_encodings(unidentified_faces, "./unidentified_faces_encodings.json")  # save the dictionary to json file
                print("Saved unidentified faces to json file.")
        print("Loaded unidentified faces.\n")


        # load identified faces
        identified_faces = {}  # {name: (image encoding [list], is safe [bool])}
        print("Loading identified faces...")
        with open("./identified_faces_encodings.json", "r") as f:
            try:
                identified_faces = json.load(f)
            except json.decoder.JSONDecodeError:
                print("Failed to load identified faces from json file.")
                print("Loading from image folder...")
                for filename in os.listdir("./identified_faces/"):  # for each image in the folder
                    image_array = face_recognition.load_image_file(("./identified_faces/" + filename))  # load image
                    image_encoding = face_recognition.face_encodings(image_array)[0].tolist()  # encode the image, assuming only one face in image
                    name = filename.split(".")[0]  # get the name of the image (filename without extension)
                    identified_faces[name] = (image_encoding, False)  # add the image encoding to the dictionary and assume each face is unsafe
                self.save_encodings(identified_faces, "./identified_faces_encodings.json")  # save the dictionary to json file
                print("Saved identified faces to json file.")
        print("Loaded identified faces.\n")
        
        return identified_faces, unidentified_faces



    def save_encodings(self, content, filename):
        """
        Saves the given encodings dictionary content to the given filename.
        """
        with open(filename, "w") as f:
            json.dump(content, f)


    def recognize_faces(self, frame):
        """
        Recognizes faces in the given frame.
        """
        # preprocess the frame for face recognition
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # resize frame to 1/4th size to reduce processing time
        frame = frame[:, :, ::-1]  # convert BGR to RGB for face_recognition library's use
        
        frame_face_locations = face_recognition.face_locations(frame)  # get face locations
        frame_face_encodings = face_recognition.face_encodings(frame, )  # get encodings for all faces in frame
        frame_faces_names = []  # list of names for all faces in frame

        identifed_faces_encodings = []
        for face_name in self.identified_faces:
            identifed_faces_encodings.append(self.identified_faces[face_name][0])

        unidentified_faces_encodings = []
        for face_name in self.unidentified_faces:
            unidentified_faces_encodings.append(self.unidentified_faces[face_name])

        for face_encoding in frame_face_encodings:
            identified_faces_matches = face_recognition.compare_faces(identifed_faces_encodings, face_encoding)  # get matches for each face in frame
            unidentified_faces_matches = face_recognition.compare_faces(unidentified_faces_encodings, face_encoding)  # get matches for each face in frame
            
            # check against identified faces
            identified_face_distances = face_recognition.face_distance(identifed_faces_encodings, face_encoding)  # get distances for each face in frame
            best_match_index = np.argmin(identified_face_distances)
            if identified_faces_matches[best_match_index] == True:
                name = list(self.identified_faces.keys())[best_match_index]
                is_safe = self.identified_faces[name][1]
                frame_faces_names.append(f"{name} - ({'safe' if is_safe else 'unsafe'})")
                continue
            
            # check against unidentified faces
            unidentified_face_distances = face_recognition.face_distance(unidentified_faces_encodings, face_encoding)  # get distances for each face in frame
            best_match_index = np.argmin(unidentified_face_distances)
            if unidentified_faces_matches[best_match_index] == True:
                id_ = list(self.unidentified_faces.keys())[best_match_index]
                frame_faces_names.append(f"{id_} - (unidentified)")
                continue

            # face is not recognized, add to unidentified faces
            id_ = str(uuid.uuid4())

                
        return frame_faces_names, frame_face_locations
                



        # best_match_index = np.argmin(face_distances)
        # if matches[best_match_index]:
        #     name = list(self.identified_faces.keys())[best_match_index]
        #     is_safe = self.identified_faces[name][1]
            
