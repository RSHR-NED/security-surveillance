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
        print("Initializing FaceRecognizer...\n\n")
        self.identified_faces, self.unidentified_faces = self.load_face_encodings()
        
        # update newest_id
        if len(self.identified_faces) + len(self.unidentified_faces) != 0:
            self.newest_id = max(max(self.identified_faces.keys()), max(self.unidentified_faces.keys()))

        print("FaceRecognizer initialized.\n\n")


    def load_face_encodings(self):
        """
        Loads the face encodings for identified and unidentified faces.
        For both, identified and unidentified faces, it first tries to load from json file.
        If that fails, it tries to load from the respective image folder.
        """

        # load unidentified faces
        unidentified_faces = {}  # {id [int]: image encoding [list]}
        print("Loading unidentified faces...")
        with open("unidentified_faces_encodings.json", "r") as f:
            try:
                unidentified_faces = json.load(f)
                unidentified_faces = {int(key): value for key, value in unidentified_faces.items()}  # convert all id keys to int
            except json.decoder.JSONDecodeError:
                print("Failed to load unidentified faces from json file.")
                print("Loading from image folder...")

                # for each image in the folder
                for filename in os.listdir("./unidentified_faces/"):
                    image_array = face_recognition.load_image_file(("./unidentified_faces/" + filename))  # load image
                    image_encoding = face_recognition.face_encodings(image_array)[0].tolist()  # encode the image, assuming only one face in image
                    id_ = self.get_newest_id()  # get a new id
                    unidentified_faces[id_] = image_encoding  # add the image encoding to the dictionary

                # save the dictionary to json file
                self.save_encodings(unidentified_faces, "./unidentified_faces_encodings.json")
                print("Saved unidentified faces to json file.")

        print("Loaded unidentified faces.\n")

        self.newest_id = int(max(unidentified_faces.keys()))  # update newest_id

        # load identified faces
        identified_faces = {}  # {id [int]: (name [str], face image encoding [list], is safe [bool])}
        print("Loading identified faces...")
        with open("./identified_faces_encodings.json", "r") as f:
            try:
                identified_faces = json.load(f)
                identified_faces = {int(key): value for key, value in identified_faces.items()}  # convert all id keys to int
            except json.decoder.JSONDecodeError:
                print("Failed to load identified faces from json file.")
                print("Loading from image folder...")

                # for each image in the folder
                for filename in os.listdir("./identified_faces/"):
                    image_array = face_recognition.load_image_file(
                        ("./identified_faces/" + filename))  # load image
                    image_encoding = face_recognition.face_encodings(image_array)[0].tolist()  # encode the image, assuming only one face in image
                    name = filename.split(".")[0]  # get the name of the image (filename without extension)
                    id_ = self.get_newest_id()  # get id for the image
                    identified_faces[id_] = (name, image_encoding, False)  # add the image encoding to the dictionary and assume each face is unsafe

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


        frame_face_locations = face_recognition.face_locations(frame)  # get locations of all faces in the frame
        frame_face_encodings = face_recognition.face_encodings(frame, frame_face_locations)  # get encodings for all faces in the frame
        frame_faces_labels = []  # list of names for all faces in frame

        identifed_faces_encodings = []
        for face_name in self.identified_faces:
            identifed_faces_encodings.append(self.identified_faces[face_name][1])

        unidentified_faces_encodings = list(self.unidentified_faces.values())

        for i, face_encoding in enumerate(frame_face_encodings):
            unidentified_faces_matches = face_recognition.compare_faces(unidentified_faces_encodings, face_encoding)  # get matches for each face in frame against all unidentified faces
            identified_faces_matches = face_recognition.compare_faces(identifed_faces_encodings, face_encoding)  # get matches for each face in frame against all identified faces
            
            # check against identified faces
            identified_face_distances = face_recognition.face_distance(identifed_faces_encodings, face_encoding)  # get distances for each face in frame
            best_match_index = np.argmin(identified_face_distances)
            if identified_faces_matches[best_match_index] == True:
                name, _, is_safe = self.identified_faces[list(self.identified_faces.keys())[best_match_index]]
                id_ = list(self.identified_faces.keys())[best_match_index]
                frame_faces_labels.append(f"{name} (id: {id_}) - {'safe' if is_safe else 'unsafe'}")
                continue  # skip to next face

            # check against unidentified faces
            unidentified_face_distances = face_recognition.face_distance(unidentified_faces_encodings, face_encoding)  # get distances for each face in frame
            best_match_index = np.argmin(unidentified_face_distances)
            if unidentified_faces_matches[best_match_index] == True:
                id_ = list(self.unidentified_faces.keys())[best_match_index]
                frame_faces_labels.append(f"Unidentified (id: {id_})")
                continue  # skip to next face

            # face is not recognized, add to unidentified faces
            new_id = self.get_newest_id()
            self.unidentified_faces[new_id] = face_encoding.tolist()
            # save encoding to json database asynchronously
            threading.Thread(
                target=self.save_encodings,
                args=(self.unidentified_faces, "./unidentified_faces_encodings.json")
                ).start()
            
            # extract face image from frame
            face_image = frame[frame_face_locations[i][0]:frame_face_locations[i][2], frame_face_locations[i][3]:frame_face_locations[i][1]]
            # save face image to folder asynchronously
            threading.Thread(
                target=cv2.imwrite,
                args=(f"./unidentified_faces/{new_id}.jpg", face_image)
                ).start()

            frame_faces_labels.append(f"Unidentified (id: {new_id})")

        # scale up frame face locations back to original size
        frame_face_locations = [(int(a * 4), int(b * 4), int(c * 4), int(d * 4)) for (a, b, c, d) in frame_face_locations]

        return frame_faces_labels, frame_face_locations


    def get_newest_id(self):
        self.newest_id += 1
        return self.newest_id
