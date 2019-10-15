import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage

filename=input("Файл скана газеты: ")
#image=np.array(object=skimage.io.imread(filename))#, as_gray=True), dtype=np.uint8)

image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#for i in range(15,240,15):
ret, image = cv2.threshold(src=image, thresh=165, maxval=255, type=cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)
#image = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

fig, ax = plt.subplots()
ax.imshow(image)


#print(image)

#for i in range(1,8,32):
#BORDER=128    
#image[image<BORDER]=0
#image[image>=BORDER]=1

#thresh, im_bw = cv2.threshold(thresh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#fig, ax = plt.subplots()
#ax.imshow(im_bw)


#print(im_bw)

image = cv2.erode(image,  kernel=np.array([[1]*5]*5), iterations = 4)
image = cv2.dilate(image, kernel=np.array([[1]*5]*5), iterations = 4)

fig, ax = plt.subplots()
ax.imshow(image)

plt.show()
