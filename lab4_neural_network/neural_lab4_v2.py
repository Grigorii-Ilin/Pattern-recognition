import pickle

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

#import matplotlib.pyplot as plt


#batch_size = 128
DIGITS_COUNT = 10
IMAGE_ROWS_COUTN, IMAGE_COLUMNS_COUNT, CHANNELS_COUNT = 28, 28, 1


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train, x_test, y_test=x_train[:1000], y_train[:1000], x_test[:1000], y_test[:1000] #FOR DEBUG ONLY

x_train = x_train.reshape(x_train.shape[0], IMAGE_ROWS_COUTN, IMAGE_COLUMNS_COUNT, CHANNELS_COUNT) # 1= канал
x_test = x_test.reshape(x_test.shape[0], IMAGE_ROWS_COUTN, IMAGE_COLUMNS_COUNT, CHANNELS_COUNT)
input_shape = (IMAGE_ROWS_COUTN, IMAGE_COLUMNS_COUNT, CHANNELS_COUNT)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

COLOR_TONES=255
x_train /= COLOR_TONES
x_test /= COLOR_TONES

print('Тренировочный набор:', x_train.shape)
print(x_train.shape[0], '(Тренировочный пример)')
print(x_test.shape[0], 'Тестовый пример')

# категориальные типы, т.е. это цифры 0-9 (список из 0000 и один 1)
y_train = keras.utils.to_categorical(y_train, DIGITS_COUNT)
y_test = keras.utils.to_categorical(y_test, DIGITS_COUNT)


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) #32= выходне признаки
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #усреднение
model.add(Dropout(0.25)) #случайно отключаем узлы
model.add(Flatten()) #сглаживание
model.add(Dense(128, activation='relu')) #полностью связанный слой
model.add(Dropout(0.5))
model.add(Dense(DIGITS_COUNT, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy']
              )


model.fit(x_train, 
         y_train,
         # batch_size=batch_size,
          epochs=2,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print("Результат", score)

fs=open(r"C:\MY_DOC_HP\BMSTU\2019_2\Pattern-recognition\lab4_neural_network\model.pkl", "wb")
pickle.dump(model.to_yaml(), fs)
fs.close()


# pred=model.predict(x_test[5:6])
# print(pred)
# plt.subplot(x_test[5])
# plt.subplot(x_test[6])
# plt.show()