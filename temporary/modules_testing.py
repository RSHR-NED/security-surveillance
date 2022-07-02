# importing modules
from detect_n_recognize.face_dnr import *
from detect_n_recognize.handle_databases import*
from time import time

# Constant

images = ImgDatabase.read_identified_faces_db("./identified_faces")
E = EncodingsDB()
E.find_encodings(images)
encodings = E.get_all_encodings()


""" # Without Modular Approach

image_M_test = fr.load_image_file('unidentified_faces/Valentin_test.jpg')
start = time()
# image_M_test = cv2.resize(image_M_test, (0,0), None, 0.25 , 0.25, interpolation=cv2.INTER_AREA)
image_M_test = cv2.cvtColor(image_M_test, cv2.COLOR_BGR2RGB)
faces = fr.face_locations(image_M_test)[0]
print(faces)
encoded= fr.face_encodings(image_M_test, [faces])[0]
print(encoded)
print(fr.compare_faces(encodings, encoded))
end = time()
print("Execution Time:", end-start) """



# With Modular Approach    

image_M_test = fr.load_image_file('unidentified_faces/Valentin_test.jpg')

start = time() 
FR = FaceRecognition()
FR.comparing_faces(image_M_test,encodings)
end = time()
print("Execution Time:", end-start)



### Advantages
# Dont need to save the images of the identified databases
# for compare faces. only on the bases of the generated 260 int values
# Hog is uses due to which slight tilt face can be detect
# 

### Research Required

# which resizing technique is best for the upsampling or down-sampling
# Resizing in the way that small face can detect
# How to set image id of the running frame
# How to optimize the algorithm 



# deep face

"""         RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to grayscale for face detection
        faces = fr.face_locations(RGB_frame)  # Detect faces

        # for every face detected
        for face in faces:
            x, y, w, h = face  # get face coordinates
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # draw rectangle around face

            face = frame[y:y+h, x:x+w]  # extract face from frame
            
            recognition = DeepFace.find(face, db_path = "./unidentified_faces")  # recognize face
            # print(recognition) """
 