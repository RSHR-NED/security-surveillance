import os
from deepface import DeepFace
import cv2

test_imgs = ["Marshmello_test.jpg", "Skrillex_test.jpg", "Slushii_test.jpg", "Valentin_test.jpg"]  # test images paths
img = cv2.imread("./test_db1/" + test_imgs[3])  # read test image
recognized = DeepFace.find(img, db_path="./test_db0", enforce_detection=False)  # run face recognition on test image
print(recognized)