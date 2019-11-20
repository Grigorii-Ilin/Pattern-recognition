
import keras
#from keras.datasets import mnist
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import keras.utils
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#%%
sns.set()


#%%
digits = load_digits() #np.array(mnist.load_data())#load_digits()
digits
digits.data

#%%
plt.plot(digits.images[0])
plt.show()
#%%

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target)#, test_size=1/5)
img_height = 8
img_width = 8
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)/16
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)/16

y_train = keras.utils.to_categorical(y_train, 10)  #3 -> [0,0,1,0,0,0,0,0,0,0]
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(8,8,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

sgd=SGD(lr=0.5) #0.01 - 0.1

model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=1, batch_size=16, epochs=12)
model.save_weights('w.h5') #load_weights(file)
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)


# %%
