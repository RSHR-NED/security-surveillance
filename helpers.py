import json
import os
import threading

import cv2
import face_recognition
import numpy as np


class FaceRecognizer:
    def __init__(self):
        """
        Initializes the FaceRecognizer.
        """
        self.newest_id = -1
        print("Initializing FaceRecognizer...\n")
        self.identified_faces, self.unidentified_faces = self.load_face_encodings()
        
        # update newest_id to the biggest ID
        if len(self.identified_faces) > 0:  # if atleast one face exists in identified faces database
            self.newest_id = max(self.identified_faces.keys())
        if len(self.unidentified_faces) > 0:  # if atleast one face exists in unidentified faces database
            self.newest_id = max(self.newest_id, max(self.unidentified_faces.keys()))

        print("FaceRecognizer initialized.\n\n")


    def load_face_encodings(self):
        """
        Loads the face encodings for identified and unidentified faces.
        For both, identified and unidentified faces, it first tries to load from json file.
        If that fails, it tries to load from the respective image folder.
        """

        # load unidentified faces
        unidentified_faces = {}  # format: {id [int]: image encoding [list]}
        print("Loading unidentified faces...")
        with open("unidentified_faces_encodings.json", "r") as f:
            # wrapping reading from json in try-except block because reading from empty json file throws json.decoder.JSONDecodeError
            try:
                unidentified_faces = json.load(f)
                unidentified_faces = {int(key): value for key, value in unidentified_faces.items()}  # convert all id keys to int (json stores all keys as str)
            except json.decoder.JSONDecodeError:
                print("Failed to load unidentified faces from json file.")
                print("Loading from image folder...")

                unidentified_faces_folder = "./unidentified_faces/"
                # for each image in the folder
                for filename in os.listdir(unidentified_faces_folder):
                    image_array = face_recognition.load_image_file((unidentified_faces_folder + filename))  # load image
                    image_encoding = face_recognition.face_encodings(image_array)[0].tolist()  # encode the image, assuming only one face in image
                    id_ = self.get_new_id()  # get a new id
                    unidentified_faces[id_] = image_encoding  # add the image encoding to the dictionary

                # save the dictionary to json file
                self.save_encodings(unidentified_faces, "./unidentified_faces_encodings.json")
                print("Saved unidentified faces to json file.")

        print("Loaded unidentified faces.\n")

        # update id to the biggest id in the dictionary if dictionary is not empty, otherwise keep it same as before
        self.newest_id = max(list(unidentified_faces.keys())) if (len(unidentified_faces) != 0) else self.newest_id

        # load identified faces
        identified_faces = {}  # {id [int]: (name [str], face image encoding [list], is safe [bool])}
        print("Loading identified faces...")
        with open("./identified_faces_encodings.json", "r") as f:
            # wrapping reading from json in try-except block because reading from empty json file throws json.decoder.JSONDecodeError
            try:
                identified_faces = json.load(f)
                identified_faces = {int(key): value for key, value in identified_faces.items()}  # convert all id keys to int
            except json.decoder.JSONDecodeError:
                print("Failed to load identified faces from json file.")
                print("Loading from image folder...")

                identified_faces_folder = "./identified_faces/"
                # for each image in the folder
                for filename in os.listdir(identified_faces_folder):
                    image_array = face_recognition.load_image_file((identified_faces_folder + filename))  # load image
                    image_encoding = face_recognition.face_encodings(image_array)[0].tolist()  # encode the image, assuming only one face in image
                    name = filename.split(".")[0]  # get the name of the image (filename without extension)
                    id_ = self.get_new_id()  # get id for the image
                    identified_faces[id_] = (name, image_encoding, True)  # add the image encoding to the dictionary and assume each face is safe

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
        param frame: the frame to recognize faces in.
        return: a tuple of face labels and corresponding locations on frame.
        """
        # preprocess the frame for face recognition
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # resize frame to 1/4th size to reduce processing time
        frame = frame[:, :, ::-1]  # convert BGR to RGB (opencv uses BGR, face_recognition uses RGB)

        # print("Detecting faces...")
        frame_face_locations = face_recognition.face_locations(frame)  # get locations of all faces in the frame
        if len(frame_face_locations) == 0:  # if no faces are detected
            print("No faces detected.")
            return [], []

        # print("Recognizing faces...")
        frame_face_encodings = face_recognition.face_encodings(frame, frame_face_locations)  # get encodings for all faces in the frame
        frame_faces_labels = []  # list of labels for all faces in the frame that can be used on the frame

        # make list of encodings for all unidentified faces
        unidentified_faces_encodings = list(self.unidentified_faces.values()) 

        # make list of encodings for all identified faces
        identifed_faces_encodings = []
        for id_ in self.identified_faces: 
            identifed_faces_encodings.append(self.identified_faces[id_][1])

        # for each face in the frame
        for i in range(len(frame_face_encodings)):
            face_encoding = frame_face_encodings[i]  # get the encoding for the current face

            unidentified_faces_matches = face_recognition.compare_faces(unidentified_faces_encodings, face_encoding)  # get matches for currrent face against all unidentified faces
            identified_faces_matches = face_recognition.compare_faces(identifed_faces_encodings, face_encoding)  # get matches for current face against all identified faces
            
            # check against identified faces
            if len(identifed_faces_encodings) != 0:  # if there are identified faces in our database
                identified_face_distances = face_recognition.face_distance(identifed_faces_encodings, face_encoding)  # get distances for each face in frame
                best_match_index = np.argmin(identified_face_distances)  # get index of the lowest distance match
                if identified_faces_matches[best_match_index] == True:  # if the lowest distance face is actually a match

                    # make label for the face
                    name, _, is_safe = self.identified_faces[list(self.identified_faces.keys())[best_match_index]]
                    id_ = list(self.identified_faces.keys())[best_match_index]
                    frame_faces_labels.append(f"{name} (id: {id_}) - {'safe' if is_safe else 'unsafe'}")
                    continue  # skip to next face

            # check against unidentified faces (to show same ID each time)
            if len(unidentified_faces_encodings) != 0:  # if there are unidentified faces in our database
                unidentified_face_distances = face_recognition.face_distance(unidentified_faces_encodings, face_encoding)  # get distances for each face in frame
                best_match_index = np.argmin(unidentified_face_distances)  # get index of the lowest distance match
                if unidentified_faces_matches[best_match_index] == True:  # if the lowest distance face is actually a match

                    # make label for the face
                    id_ = list(self.unidentified_faces.keys())[best_match_index]
                    frame_faces_labels.append(f"Unidentified (id: {id_})")
                    continue  # skip to next face
            
            
            
            # face is not recognized as any existing identified or unidentified face, add to unidentified faces
            new_id = self.get_new_id()
            self.unidentified_faces[new_id] = face_encoding.tolist()
            threading.Thread(
                target=self.save_encodings,
                args=(self.unidentified_faces, "./unidentified_faces_encodings.json")
                ).start()  # save encoding to json database asynchronously
            
            self.identified_faces, self.unidentified_faces = self.load_face_encodings()
            
            # extract face image from frame
            face_location = frame_face_locations[i]
            face_image = frame[face_location[0]:face_location[2], face_location[3]:face_location[1]]
            face_image = cv2.resize(face_image, (0, 0), fx=4, fy=4)
            face_image = face_image[:, :, ::-1]  # convert from RGB to BGR (which CV2 understands) for saving to image folder
            threading.Thread(
                target=cv2.imwrite,
                args=(f"./unidentified_faces/{new_id}.jpg", face_image)
                ).start()  # save face image to folder asynchronously

            # label for the face
            frame_faces_labels.append(f"Unidentified (id: {new_id})")

        # scale up frame face locations back to original size
        frame_face_locations = [(int(a * 4), int(b * 4), int(c * 4), int(d * 4)) for (a, b, c, d) in frame_face_locations]  

        print(f"Recognized {len(frame_faces_labels)} faces.")
        return frame_faces_labels, frame_face_locations


    def get_new_id(self):
        self.newest_id += 1
        return self.newest_id