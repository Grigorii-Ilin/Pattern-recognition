import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import cv2

image=np.array(skimage.data.coins())
#print(image)
fig, ax = plt.subplots()
ax.imshow(image, cmap='Greys_r')


image[image<128]=0
image[image>=128]=1
fig, ax = plt.subplots()
ax.imshow(image, cmap='Greys_r')

plt.show()

input()