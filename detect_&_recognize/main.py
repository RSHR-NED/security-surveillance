from face_dnr import *
from handle_databases import *


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