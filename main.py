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
    output = Dense(13, activation='sigmoid')(class1) # Ostatnia warstwa musi miec tyle wyjsc ile klas obrazow

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
    train_it = datagen.flow_from_directory('new_train/', batch_size=3, target_size=(32, 32))

    # Trenowanie sieci neuronowej. Podajemy ilosc epok i dlugosc krokow na kazda z nich
    model.fit(train_it, steps_per_epoch=len(train_it), epochs=6)

    # Zapisanie modelu do pamieci, tak aby moc korzystac z niego pozniej
    model.save('model_znakow.h5')


def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(32, 32))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


# Klasyfikacja obrazu
def testing():
    model = load_model('model_znakow.h5')
    #nazwy_znakow = ["masz_pierwszenstwo", "nakaz_lewo", "nakaz_prawo", "nakaz_prosto",
    #                "ostry_prawo", "stop", "ustap", "zakaz_ruchu", "zakaz_wajzdu", "zakaz_wyprzedzania"]

    nazwy_znakow = ["animal", "left", "priority", "red-right","blue-right", "roboty",
                    "rondo", "snow", "stop", "ustap", "zakaz_tirow", "zakaz-ruchu", "zakaz-wjazdu"]
    # OBRAZ

    #image = load_image('video/00034_00011_00027.png')

    # VIDEO
    count = 0
    vidcap = cv2.VideoCapture('new_film_1.mp4')
    success,image = vidcap.read()

    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*500))    # Tutaj mozna wybrac czestosc probkowania
        success,image = vidcap.read()

        # Gdyby wideo bylo w zlym formacie to sa ponizsze funkcje zeby program sie nie wysypal.
        # Ale zeby dzialalo dobrze, to wszystko w tej samej rozdzielczosci musi byc.
        image = cv2.resize(image, (32, 32))
        image = image.reshape(1, 32, 32, 3)

        # Najwazniejsza czesc do przewidywania
        # result -> To tablica o liczbie miejsc rownej liczbie klas.
        # Poszczegolne wartosci odpowiadaja prawdopodobienstwu danego obrazu
        result = model.predict(image)

        # Szukanie najwiekszej wartosci w tablicy - najbardziej prawdopodobnego obrazu
        index_max = np.argmax(result)
        value_max = np.max(result)

        # Wypisanie na konsoli co zostalo wykryte. Jesli male prawdopodobienstwo to wypisuje ze nic nie pasuje
        if (value_max > 0.8):
            print("Wykryto: " + nazwy_znakow[index_max])
        else:
            print("Nic mi nie pasuje... :-(")
            print(result)
        # Zwiekszamy licznik i wracamy do wideo
        count = count + 1


# Wywolanie odpowiednich funkcji do trenowania lub klasyfikacji.
# Odkomentowac odpowiednie:

#training()
testing()