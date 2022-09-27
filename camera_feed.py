import cv2
from helpers import FaceRecognizer


class VideoStream():
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.face_recognizer = FaceRecognizer()

    def __del__(self):
        self.cam.release()

    def get_frame(self):
        # read a frame
        ret, frame = self.cam.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            return None

         # mirror frame horizontally
        frame = cv2.flip(frame, 1) 

        # perform face recognition
        # self.mark_faces(frame)

        ret, image = cv2.imencode('.jpg', frame)
        return image.tobytes()


    def mark_faces(self, frame):
        # recognize faces in the frame
        faces_labels, face_locations = self.face_recognizer.recognize_faces(frame)

        # recognize faces in the frame
        frame_faces_labels, frame_face_locations = self.face_recognizer.recognize_faces(frame)

        # draw rectangles around faces
        for (top, right, bottom, left), label in zip(frame_face_locations, frame_faces_labels):
            # draw box and label around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)  
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
