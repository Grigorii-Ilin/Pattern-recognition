import cv2
import matplotlib.pyplot as plt
import glob

work_dir=r"C:\MY_DOC_HP\BMSTU\2019_2\Pattern-recognition\adobe_panoramas\data"

orig_image = cv2.imread(work_dir+r"\carmel-00.png")
orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
orig_key_points, orig_descriptions = sift.detectAndCompute(orig_image, None)
orig_img_with_key_points = cv2.drawKeypoints(orig_image, orig_key_points, None)
plt.imshow(orig_img_with_key_points)
plt.show()

best_matches_count=0
best_file=""
figure=plt.figure(figsize=(2,2))
for filename in glob.glob(work_dir+r"\*\*.png"):

    cur_img = cv2.imread(filename)
    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    cur_key_points, cur_descriptions = sift.detectAndCompute(cur_img, None)
    knn_match = cv2.BFMatcher().knnMatch(orig_descriptions, cur_descriptions, k=2)

    good_matches=[m1 for m1, m2 in knn_match if m1.distance<m2.distance/2]
    cur_matches_count=len(good_matches)

    if cur_matches_count >= 10:
        draw_matches=cv2.drawMatches(orig_image, 
                                    orig_key_points,
                                    cur_img, 
                                    cur_key_points,
                                    good_matches,
                                    None)
               
        figure.add_subplot(2,2,1)
        plt.imshow(orig_image)
        figure.add_subplot(2,2,2)
        plt.imshow(cur_img)
        figure.add_subplot(2,2,3)
        plt.imshow(draw_matches)
        plt.show()

    if cur_matches_count>best_matches_count:
        best_matches_count=cur_matches_count
        best_file=filename

    print(filename, cur_matches_count)

print("\nЛучшее совпадение:", best_file, best_matches_count)