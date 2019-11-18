import numpy as np
import matplotlib.pyplot as plt
import skimage.io#.data
import cv2
#from PIL.Image import open

filename=input("Файл скана газеты: ")
#image=np.array(object=skimage.data.coins(), dtype=np.uint8)
image=np.array(object=skimage.io.imread(filename), dtype=np.uint8)
#image = cv2.imread(filename)
#image=skimage.io.imread(filename)
#=np.array(image, dtype=np.uint8)
print(image)

fig, ax = plt.subplots()
ax.imshow(image, cmap='Greys_r')

BORDER=200

image=np.average(image, axis=2)
image[image<BORDER]=0
image[image>=BORDER]=1
fig, ax = plt.subplots()
ax.imshow(image, cmap='Greys_r')

img2=cv2.morphologyEx(src=image, op=cv2.MORPH_CLOSE, kernel=np.array([[1]*9]*9))
fig, ax = plt.subplots()
ax.imshow(img2, cmap='Greys_r')

# img3=cv2.dilate(src=img2, kernel=np.array([[1]*9]*9))
# fig, ax = plt.subplots()
# ax.imshow(img3, cmap='Greys_r')

img3=cv2.erode(src=img2, kernel=np.array([[1]*9]*9))
fig, ax = plt.subplots()
ax.imshow(img3, cmap='Greys_r')

img3=cv2.dilate(src=img2, kernel=np.array([[1]*9]*9))
fig, ax = plt.subplots()
ax.imshow(img3, cmap='Greys_r')

plt.show()

#input()