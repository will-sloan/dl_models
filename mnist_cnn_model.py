from keras.datasets import mnist
import time
from numpy import mean, std
from keras.utils import np_utils
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Convolution2D, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
import numpy as np
start_time = time.time()
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Changing all values to between 0 and 1
x_train = x_train/255.0
x_test = x_test/255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]
def build_cnn():
    classifier = Sequential()
    classifier.add(Convolution2D(128, (3,3), padding='same', input_shape=(28,28,1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.2))
    classifier.add(Convolution2D(64, (3,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    # classifier.add(Dropout(0.2))
    # classifier.add(Convolution2D(32, (3,3), activation='relu'))
    # classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=32, activation='relu'))
    # classifier.add(Dropout(0.5))
    # classifier.add(Dense(units=32, activation='relu'))
    classifier.add(Dense(units=num_classes, activation='softmax'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

def kfold_train(model, datax, datay):
    scores = list()
    kfold = KFold()
    for i_train, i_test in kfold.split(datax):
        #Two vairables created are the indexes, so they can be used on both y and x
        x_train, y_train, x_test, y_test = datax[i_train], datay[i_train], datax[i_test], datay[i_test]
        model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
        _, acc = model.evaluate(x_test, y_test)
        scores.append(acc)
    return scores


def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))


classifier = build_cnn()
#classifier.fit(x_train, y_train, epochs=100, batch_size=40, validation_data=(x_test,y_test))
#classifier.evaluate(x_test, y_test)
scores = kfold_train(classifier, x_train, y_train)
summarize_performance(scores)
print(time.time()-start_time)
