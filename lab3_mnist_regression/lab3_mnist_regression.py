import pickle
from string import digits

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
# digits = load_digits()
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='.')
#Получить набор данных из openml по имени или идентификатору набора данных.
# Наборы данных однозначно идентифицируются либо целочисленным идентификатором, либо сочетанием имени и версии
# (т. Е. Может быть несколько версий набора данных «iris»). Пожалуйста, дайте имя или data_id (не оба).
# В случае, если имя дано, версия также может быть предоставлена.
# plt.imshow(np.reshape(digits.data[60], (8,8)), cmap='Greys_r')
# plt.show()

print(X.data.shape)
# X=X[:1000]
# y=y[:1000]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
#Разбить массивы или матрицы на случайные наборы поездов и тестов
#Быстрая утилита, которая объединяет проверку входных данных и приложение
# для ввода данных в один вызов для разделения (и, возможно, дополнительной выборки) данных в oneliner.next(ShuffleSplit().split(X, y))
logr = LogisticRegression(solver="lbfgs") #because warning
#Классификатор логистической регрессии

logr.fit(x_train, y_train)
#Установите модель в соответствии с данными тренировки.
digit = x_test[0]
print(logr.predict([digit]))
#Предсказать метки класса для образцов в X
plt.imshow(np.reshape(digit, (28, 28)), cmap='Greys_r')
plt.show()

print(logr.score(x_test, y_test))
#Возвращает среднюю точность данных и данных теста
pickle.dump(logr, open('logr.pkl', 'wb'))
#Записывает сериализованный объект в файл.