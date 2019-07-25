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
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import operator
from tensorflow.keras.preprocessing.image import ImageDataGenerator




def prep_data(x_train, x_test, y_train, y_test):
    '''
    This prep is required to grab only photos with cars(1), cats(3), dogs(5), and trucks(9)
    Through this boring and probably unnecessarily slow process, only these 4 categories are grabbed
    '''
    #Put together photos with their answers
    dataset = list(zip(x_test, y_test))
    #Sort them by their answers
    dataset.sort(key=operator.itemgetter(1))
    #Put back into seperate lists
    x_test, y_test = list(zip(*dataset))
    #Take out only cars, cats, dogs, and trucks
    good_test_x =  np.array(x_test[1000:2000] + x_test[3000:4000] + x_test[5000:6000] + x_test[9000: 10000])
    good_test_y = y_test[1000:2000] + y_test[3000:4000] + y_test[5000:6000] + y_test[9000: 10000]
    #Replace complete set with new reduced set
    x_test, y_test = good_test_x, good_test_y
    #Redo for the training data
    dataset = list(zip(x_train, y_train))
    dataset.sort(key=operator.itemgetter(1))
    x_train, y_train = list(zip(*dataset))
    good_dataset_x =  np.array(x_train[5000:10000] + x_train[15000:20000] + x_train[25000:30000] + x_train[45000: 50000])
    good_dataset_y = y_train[5000:10000] + y_train[15000:20000] + y_train[25000:30000] + y_train[45000: 50000]
    x_train, y_train = good_dataset_x, good_dataset_y

    #Change to numpy array to preform replace each number to its lower value (ei dogs were 5 but now are 3)
    y_train = np.array(y_train)
    y_train = np.where(y_train==1, 0, y_train)
    y_train = np.where(y_train==3, 1, y_train)
    y_train = np.where(y_train==5, 2, y_train)
    y_train = np.where(y_train==9, 3, y_train)
    y_train = y_train.tolist()
    y_train = list(zip(*y_train))[0]

    y_test = np.array(y_test)
    y_test = np.where(y_test==1, 0, y_test)
    y_test = np.where(y_test==3, 1, y_test)
    y_test = np.where(y_test==5, 2, y_test)
    y_test = np.where(y_test==9, 3, y_test)
    y_test = y_test.tolist()
    y_test = list(zip(*y_test))[0]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    return x_train, x_test, y_train, y_test

def data_aug(datax, datay, testx, testy, bs=32):
    datagen = ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
    x = datagen.flow(datax, datay, batch_size=bs)
    y = datagen.flow(testx, testy, batch_size=bs)
    return x, y

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
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=4, activation='softmax', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

def train_model(model, datax, datay, n_folds=3, bs=32, epochs=25):
    scores = list()
    kfold = KFold(n_folds, shuffle=True)

    for i_train, i_test in kfold.split(datax):
        x_train, y_train, x_test, y_test = datax[i_train], datay[i_train], datax[i_test], datay[i_test]
        x_train, x_test  = data_aug(x_train, y_train, x_test, y_test, bs)
        model.fit_generator(x_train, steps_per_epoch=20000/bs, epochs=epochs, validation_data= x_test, validation_steps=4000/bs)
    return model

def summary_scores(scores):
    print(f'Total number of scores {len(scores)} have an mean of {mean(scores)}')

def main():
    scores = list()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test, y_train, y_test = prep_data(x_train, x_test, y_train, y_test)
    model = build_model()
    model = train_model(model,x_train, y_train, n_folds=5, bs=48, epochs=50)
    _, acc = model.evaluate(x_test, y_test)
    print(f'Final Test acc is {acc}')

main()
