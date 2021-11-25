import cv2
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

'''
PROGRAM DO TRENOWANIA SIECI NEURONOWEJ I KLASYFIKACJI OBRAZOW
Gdyby nie chciało działać, trzeba sprawdzic, zeby wersje kerasa i tensorflowa byly jednakowe.
Nie przejmowac sie czerwonymi napisami w konsoli, tak ma byc xD

'''

# Trenowanie sieci
def training():
    # Model VGG16 sluzy do klasyfikacji obrazow bo zawiera w sobie dodatkowe warstwy
    model = VGG16(include_top=False, input_shape=(32, 32, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False

    # Dodawanie nowych warstw sieci neuronowej
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(6, activation='sigmoid')(class1) # Ostatnia warstwa musi miec tyle wyjsc ile klas obrazow

    # Wyswietlanie w konsoli jaki model zostal stworzony
    model.summary()

    # Definicja modelu
    model = Model(inputs=model.inputs, outputs=output)

    # Kompilacja modelu
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    train_it = datagen.flow_from_directory('train_22_11/', batch_size=6, target_size=(32, 32))
    test_it = datagen.flow_from_directory('test_22_11/', batch_size=6, target_size=(32, 32))
    # Trenowanie sieci neuronowej. Podajemy ilosc epok i dlugosc krokow na kazda z nich
    model.fit(train_it, steps_per_epoch=len(train_it), epochs=4, validation_data=test_it, validation_steps=len(test_it))

    # Zapisanie modelu do pamieci, tak aby moc korzystac z niego pozniej
    model.save('model_22_11.h5')

training()