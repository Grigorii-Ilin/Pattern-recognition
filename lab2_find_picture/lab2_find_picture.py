import cv2
import matplotlib.pyplot as plt
#import numpy
import glob#, os

work_dir=r"C:\MY_DOC_HP\BMSTU\2019_2\Pattern-recognition\adobe_panoramas\data"

orig_image = cv2.imread(work_dir+r"\carmel-00.png")
orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
orig_key_points, orig_descriptions = sift.detectAndCompute(orig_image, None)
orig_img_with_key_points = cv2.drawKeypoints(orig_image, orig_key_points, None)
plt.imshow(orig_img_with_key_points)
plt.show()

#os.chdir(r"C:\MY_DOC_HP\BMSTU\2019_2\Pattern-recognition\adobe_panoramas\data\carmel")
best_matches_count=0
best_file=""
for filename in glob.glob(work_dir+r"\*\*.png"):

    cur_img = cv2.imread(filename)
    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    #(kp, des) = sift.detectAndCompute(image, None)
    cur_key_points, cur_descriptions = sift.detectAndCompute(cur_img, None)
    #i = cv2.drawKeypoints(orig_image, orig_key_points, None)
    knn_match = cv2.BFMatcher().knnMatch(orig_descriptions, cur_descriptions, k=2)
    #matches = match.knnMatch(orig_descriptions, cur_descriptions, k=2)
    #good_matches = numpy.array([m1 for m1, m2 in matches if m1.distance < m2.distance / 2])

    cur_matches_count=len([True for m1, m2 in knn_match if m1.distance<m2.distance/2])

    # img3 = cv2.drawMatches(image, kp, image2, kp2, good_matches, None)
    # if good_matches.size > 100:
    #    # plt.imshow(img3)
    #    # plt.show()
    #    image = image2

    #    print(file, good_matches.size)

    if cur_matches_count>best_matches_count:
        best_matches_count=cur_matches_count
        best_file=filename

    print(filename, cur_matches_count)

print("Лучшее совпадение:", best_file, best_matches_count)