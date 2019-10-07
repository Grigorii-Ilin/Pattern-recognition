import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import cv2

image=np.array(object=skimage.data.coins(), dtype=np.uint8)
fig, ax = plt.subplots()
ax.imshow(image)#, cmap='Greys_r')


image[image<128]=0
image[image>=128]=1
fig, ax = plt.subplots()
ax.imshow(image, cmap='Greys_r')

img2=cv2.morphologyEx(src=image, op=cv2.MORPH_CLOSE, kernel=np.array([[1]*9]*9))
fig, ax = plt.subplots()
ax.imshow(img2, cmap='Greys_r')

img3=cv2.erode(src=img2, kernel=np.array([[1]*3]*3))
fig, ax = plt.subplots()
ax.imshow(img3, cmap='Greys_r')

plt.show()

input()