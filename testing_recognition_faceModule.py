import numpy as np
import cv2
import os
import face_recognition as fr


def test_recognition_face():
    # Load the jpg file into a numpy array
    test_imgs = ["Marshmello_test.jpg", "Skrillex_test.jpg", "Slushii_test.png", "Valentin_test.jpg"]  # test images paths
    image_M = fr.load_image_file("test_db1/" + test_imgs[3])  # read test image in array
    image_M = cv2.cvtColor(image_M, cv2.COLOR_BGR2RGB)
    face_loc_M = fr.face_locations(image_M)[0]
    face_enc_M = fr.face_encodings(image_M)[0]


    image_M_test = fr.load_image_file('test_db0/Valentin0.jpg')
    image_M_test = cv2.cvtColor(image_M_test, cv2.COLOR_BGR2RGB)
    face_loc_M_test = fr.face_locations(image_M_test)[0]
    face_enc_M_test = fr.face_encodings(image_M_test)[0]

    cv2.rectangle(image_M, (face_loc_M[3], face_loc_M[0]), (face_loc_M[1], face_loc_M[2]), (255, 0, 0), 2)
    cv2.rectangle(image_M_test, (face_loc_M_test[3], face_loc_M_test[0]), (face_loc_M_test[1], face_loc_M_test[2]), (255, 0, 0), 2)
    cv2.imshow('Marshmello', image_M)
    cv2.imshow('Marshmello_test', image_M_test)



    results = fr.compare_faces([face_enc_M], face_enc_M_test)
    print(results)

    print(face_loc_M)
    cv2.waitKey(0)




test_recognition_face()
    

