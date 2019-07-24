'''
Goal of this project is develop a cnn model to correctly identify cars, trucks, cat and dogs
The reason for seperating out these types from all the others is to get practice manipulating datasets
Once done the model, it will be used with a raspberry pi + camera to identify any visitors outside

This model is going to be the same as my cifar-10 model but will be trained on less data
So why do I not use the other? I'm not sure but I'll be using a RPi later so it may not be as slow
'''

from numpy import mean
import sys
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


#Better Printing for debugging

np.set_printoptions(threshold=sys.maxsize)


def prep_data(x_train, x_test, y_train, y_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, x_test, y_train, y_test

#print(y_train[1])

def build_model():
    classifier = Sequential()
    classifier.add(Conv2D(64, (3,3), padding='same', input_shape=(32,32,3), activation='relu'))
    classifier.add(MaxPooling2D((2,2), padding='same'))
    classifier.add(Dropout(0.5))
    classifier.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    classifier.add(MaxPooling2D((2,2), padding='same'))
    classifier.add(Dropout(0.5))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=10, activation='softmax', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

def train_model(model, datax, datay):
    scores = list()
    kfold = KFold()
    metrics_names = model.metrics_names
    for i_train, i_test in kfold.split(datax):
        x_train, y_train, x_test, y_test = datax[i_train], datay[i_train], datax[i_test], datay[i_test]
        model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_test, y_test),
                    shuffle=True)
        _,acc = model.evaluate(x_test, y_test)
        scores.append(acc)
    print(f'\n\n{metrics_names}\n\n')
    return scores, model

def summary_scores(scores):
    print(f'Total number of scores {len(scores)} have an mean of {mean(scores)}')

def main():
    scores = list()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test, y_train, y_test = prep_data(x_train, x_test, y_train, y_test)
    model = build_model()
    scores, model = train_model(model,x_train, y_train)
    summary_scores(scores)
    _, acc = model.evaluate(x_test, y_test)
    print(f'Final Test acc is {acc}')
main()
