import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml


PKL_FILENAME='logr.pkl'

def logistic_regression(need_to_train=True):
        
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='.')
 

    print(X.data.shape)
    # X=X[:1000]
    # y=y[:1000]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
    logr = LogisticRegression(solver="lbfgs") #warning
    #Классификатор логистической регрессии

    if need_to_train:
            
        logr.fit(x_train, y_train)
        #Установить модель в соответствии с данными тренировки.
        x = x_test[0]
        print(logr.predict([x]))
        #Предсказать метки класса для образцов в X
        plt.imshow(np.reshape(x, (28, 28)), cmap='Greys_r')
        plt.show()

        print(logr.score(x_test, y_test))
        #Возвращает среднюю точность данных и данных теста

        with open(PKL_FILENAME, 'wb') as f:
            pickle.dump(logr, f)
        #Записывает сериализованный объект в файл.

    else:

        with open(PKL_FILENAME, 'rb') as f:
            logr=pickle.load(f)

      
        errors_and_real = []
        for x, y in zip(x_test, y_test):
            if logr.predict([x]) != y: 
                errors_and_real.append([x, y])
                if len(errors_and_real)>=5:
                    break
                

        plt.figure(figsize=(20,4))
        for plot_index, err in enumerate(errors_and_real):
            plt.subplot(1, 5, plot_index + 1)
            plt.imshow(np.reshape(err[0], (28,28)), cmap='Greys_r')
            plt.title("Предск: {}, Реал: {}".format(logr.predict([err[0]]), err[1]), fontsize = 13)

        plt.show()


answer=input("Обучить методом логистрической регрессии заново? Y/N ")

is_need_to_train=answer=="Y"
logistic_regression(is_need_to_train)
