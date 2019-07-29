#Build CNN:
#Step 1: Convolution
#Step 2: Max MaxPooling
#Step 3: Flattening
#Step 4: Full Connection
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Convolution2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import date
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np

print(tf.__version__)
#print(tf.test.is_gpu_available())
def build_classifier():
    classifier = Sequential()

    classifier.add(Convolution2D(64, (4, 4), input_shape=(64, 64, 3), activation= 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(32, (4, 4), activation= 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(64, (4, 4), activation= 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Flatten())

    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


def train_cnn(width=64, height=64, bs=32, epochs=100):
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
            '/home/will/Desktop/Programming/dl-learning-code/P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/training_set',
            target_size=(width, height),
            batch_size=bs,
            class_mode='binary')

    test_set = test_datagen.flow_from_directory(
            '/home/will/Desktop/Programming/dl-learning-code/P16-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/test_set',
            target_size=(width, height),
            batch_size=bs,
            class_mode='binary')

    classifier = build_classifier()

    classifier.fit_generator(
            training_set,
            steps_per_epoch=8000/bs,
            epochs=epochs,
            validation_data=test_set,
            validation_steps=2000/bs)

    classifier.save(f'cnn_{bs}_{epochs}_{date.today()}.h5')
    x = load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    prediction = classifier.predict(x)
    with open('cnn_data_storage.txt', 'a') as file:
        file.write(f'\nOn the day of {date.today()}, the format is {training_set.class_indices} and the model predicted {prediction} for a dog photo\n')

train_cnn(epochs=10)
