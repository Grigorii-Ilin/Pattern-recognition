import cv2
import matplotlib.pyplot as plt
import numpy
import glob, os

image = cv2.imread(r"C:\MY_DOC_HP\BMSTU\2019_2\Pattern-recognition\adobe_panoramas\data\carmel-00.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
(kp, des) = sift.detectAndCompute(image, None)
i = cv2.drawKeypoints(image, kp, None)
plt.imshow(i)
plt.show()

os.chdir(r"C:\MY_DOC_HP\BMSTU\2019_2\Pattern-recognition\adobe_panoramas\data\carmel")
for file in glob.glob("*.png"):

    image2 = cv2.imread(file)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    #(kp, des) = sift.detectAndCompute(image, None)
    (kp2, des2) = sift.detectAndCompute(image2, None)
    i = cv2.drawKeypoints(image, kp, None)
    match = cv2.BFMatcher()
    matches = match.knnMatch(des, des2, k=2)
    good_matches = numpy.array([m1 for m1, m2 in matches if m1.distance < m2.distance / 2])

    img3 = cv2.drawMatches(image, kp, image2, kp2, good_matches, None)
    if good_matches.size > 100:
       # plt.imshow(img3)
       # plt.show()
       image = image2

       print(file, good_matches.size)