import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage

filename=input("Файл скана газеты: ")
image=np.array(object=skimage.io.imread(filename), dtype=np.uint8)
fig, ax = plt.subplots()
ax.imshow(image)

plt.show()