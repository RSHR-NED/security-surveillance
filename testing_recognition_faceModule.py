import numpy as np
import cv2
import os
import face_recognition as fr


def test_recognition_face():
    # Load the jpg file into a numpy array
    image_M = fr.load_image_file("identified_faces/1.png")  # read test image in array
    image_M = cv2.cvtColor(image_M, cv2.COLOR_BGR2RGB)
    resized_image_M = cv2.resize(image_M, (0,0), None, 0.25 , 0.25, interpolation=cv2.INTER_AREA)
    cv2.imshow("pic", resized_image_M)
    
    cv2.waitKey(0)
    input()
    face_loc_M = fr.face_locations(image_M)[0]
    face_enc_M = fr.face_encodings(image_M)[0]
    print(face_enc_M)
    print(type(face_enc_M))
    print(len(face_enc_M))
    input()


    image_M_test = fr.load_image_file('unidentified_faces/Marshmello_test.jpg')
    image_M_test = cv2.cvtColor(image_M_test, cv2.COLOR_BGR2RGB)
    face_loc_M_test = fr.face_locations(image_M_test)[0]
    face_enc_M_test = fr.face_encodings(image_M_test)[0]

    cv2.rectangle(image_M, (face_loc_M[3], face_loc_M[0]), (face_loc_M[1], face_loc_M[2]), (255, 0, 0), 2)
    cv2.rectangle(image_M_test, (face_loc_M_test[3], face_loc_M_test[0]), (face_loc_M_test[1], face_loc_M_test[2]), (255, 0, 0), 2)
    cv2.imshow('Marshmello', image_M)
    cv2.imshow('Marshmello_test', image_M_test)



    results = fr.compare_faces([face_enc_M], face_enc_M_test)
    cv2.waitKey(0)




test_recognition_face()
    

