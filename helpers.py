import face_recognition
import json
import os


class FaceRecognizer:
    def __init__(self):
        """
        Initializes the FaceRecognizer.
        """
        print("Initializing FaceRecognizer...\n\n")
        self.identified_faces, self.unidentified_faces = self.load_face_encodings()
        


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
                    name = filename.split(".")[0]  # get the name of the image (filename without extension)
                    unidentified_faces[name] = image_encoding  # add the image encoding to the dictionary
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

    

