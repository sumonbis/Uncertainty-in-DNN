#!/bin/python
# -*- coding: utf8 -*-

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28


def input_data(percent):
    # the data, split between train and test sets
    dataset = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = dataset
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    x_train = x_train[:int(x_train.shape[0] * percent / 100)]
    y_train = y_train[:int(y_train.shape[0] * percent / 100)]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    return x_train, y_train, x_test, y_test, input_shape


def create_model(hidden_unit):
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(hidden_unit, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    return model


def train(model, x_train, y_train, x_test, y_test, epoch):
    print("\nTraining ...")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(x_test, y_test))
    return model

def evaluate(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    #return score[0], score[1]
    return score[1]

def train_and_evaluate(percent, epoch, hidden_unit):
    x_train, y_train, x_test, y_test, input_shape = input_data(percent)
    dnn_model = create_model(hidden_unit)
    model = train(dnn_model, x_train, y_train, x_test, y_test, epoch)
    return evaluate(model, x_test, y_test)

def main():
    acc = train_and_evaluate(10, 1, 64)
    print(acc)

if __name__ == "__main__":
    main()