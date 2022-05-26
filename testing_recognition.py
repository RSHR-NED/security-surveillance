import os
from deepface import DeepFace
import cv2
from time import time


test_imgs = ["Marshmello_test.jpg", "Skrillex_test.jpg", "Slushii_test.jpg", "Valentin_test.jpg"]  # test images paths
img = cv2.imread("./test_db1/" + test_imgs[3])  # read test image
start = time()
print("Starting Recognition")
recognized = DeepFace.find(img, db_path="./test_db0", enforce_detection=False)  # run face recognition on test image
print("Ending Recoginition")
end = time()
exec_time = end - start
print()
print("Execution Time:", exec_time)
print()
print(recognized)
input()
