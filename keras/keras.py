import keras
from sklearn.dataset import load_digits
from sklearn.model_selection import train_test_split

import keras.utils
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizer import SGD
from keras.losses import categorica_crossentropy

digits=load_digits()
x_train, x_test, y_train, y_test= train_test_split(digits.data, digits.target, test_size=1/5)

img_heght=8 #28
img_width=8 #28\

x_train=x_train.reshape(x_train.shape[0], img_heght, img_width, 1)
x_test=x_train.reshape(x_train.shape[0], img_heght, img_width, 1)

y_train=keras.utils.to_categorical(y_train, 10) #3 ->[0,0,1,0,0,0,0,0,0]
y_test=keras.utils.to_categorical(y_train, 10) 

model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

sgd=SGD(lr=0.5) #begin from 0.01-0.1 - lambda
model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=1, batch_size=16, epochs=2)
#batch- gradient after 16 img

model.save_weights('w.h5')

loss, accuracy=model.evaluate(x_test, y_test)
print(accuracy)