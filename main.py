import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.models import load_model

# rozdzielczość kamery
frameWidth = 640
frameHeight = 480

# inicjalizacja kamerki
cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass

def testing(image):
    model = load_model('model_22_11.h5')
    nazwy_znakow = ["priority", "roboty", "rondo", "stop", "ustap", "zakaz-wjazdu"]
    # OBRAZ


    # Gdyby wideo bylo w zlym formacie to sa ponizsze funkcje zeby program sie nie wysypal.
    # Ale zeby dzialalo dobrze, to wszystko w tej samej rozdzielczosci musi byc.
    image = cv.resize(image, (32, 32))
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
        print(result)
    else:
        print("Nic mi nie pasuje... :-(")



# funkcja do wyświetlania wielu obrazów w jednym oknie
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None,
                                               scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# funkcja do znajdywania oraz rysowania konturów na obrazie
def getContours(img, imgContour):

    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        # Ustawianie pola powierzchni, ktore ma byc wykrywane
        areaMin = 11000
        if area > areaMin:
            #cv.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv.boundingRect(approx)

            # Wycinanie regionu zainteresowania
            ROI = imgContour[y:y + h, x:x + w]
            ROIresized = cv.resize(ROI, (100, 100))

            testing(ROIresized)
            cv.imshow("Wykryte", ROIresized)


# główna pętla w której dzieje się przetwarzanie obrazu
while True:
    success, img = cap.read()
    imgContour = img.copy()
    # zblurowanie obrazu
    imgBlur = cv.GaussianBlur(img, (7, 7), 1)
    # zmiana palety barw na odcienie szarości
    imgGray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)

    # parametr do zmiany w okienku
    threshold1 = 180
    # parametr do zmiany w okienku
    threshold2 = 180
    # wykrywanie konturów za pomocą operacji Canny
    imgCanny = cv.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, imgContour)

    # inicjalizacja wyświetlanych operacji
    imgStack = stackImages(0.8, ([img, imgGray, imgCanny], [imgDil, imgContour, imgContour]))

    cv.imshow("Result", imgStack)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break