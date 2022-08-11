# python read_image.py model labelsLB.dat test1.png

import sys
import pickle
import cv2 as cv
import numpy as np
from keras.models import load_model

def getRows(image):
    # Thresholding
    bitImage = ~cv.threshold(image, 0, 255, cv.THRESH_OTSU)[1]

    # Dilating text horizontally
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20,3))
    dilatedImage = cv.dilate(bitImage, kernel, iterations=2)

    # Getting row coordinates and sorting by Y axis
    contours = cv.findContours(dilatedImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    rows = []
    for contour in contours:
        rows.append(cv.boundingRect(contour))
    rows = sorted(rows, key=lambda y: y[1])

    rowImages = []
    for row in rows:
        x, y, w, h = row
        rowImages.append(image[y:y + h, x:x + w])

    return rowImages


def getWords(rowImage):
    # Thresholding
    bitImage = ~cv.threshold(rowImage, 0, 255, cv.THRESH_OTSU)[1]

    # Dilating text horizontally
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (8,3))
    dilatedImage = cv.dilate(bitImage, kernel, iterations=1)

    # Getting word coordinates and sorting by X axis
    contours = cv.findContours(dilatedImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    words = []
    for contour in contours:
        words.append(cv.boundingRect(contour))
    words = sorted(words, key=lambda x: x[0])

    wordImages = []
    for word in words:
        x, y, w, h = word
        wordImages.append(rowImage[y:y + h, x:x + w])

    return wordImages


def getChars(wordImage):
    # Thresholding
    bitImage = ~cv.adaptiveThreshold(wordImage,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,3,22)

    # Dilating text vertically
    kernel = np.array(
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]], np.uint8)
    dilatedImage = cv.dilate(bitImage, kernel, iterations=2)

    # Getting characters and sorting by X axis
    contours = cv.findContours(dilatedImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    chars = []
    for contour in contours:
        chars.append(cv.boundingRect(contour))
    chars = sorted(chars, key=lambda x: x[0])

    string = ''
    for char in chars:
        x, y, w, h = char
        
        # Adding margin to create a square image and resizing
        if h > w:
            charImage = cv.copyMakeBorder(wordImage[y:y + h, x:x + w], 
                top=None, bottom=None, left=(h-w)//2, right=-((h-w)//-2), 
                borderType=cv.BORDER_CONSTANT, value=[255,255,255])
        elif w > h:
            charImage = cv.copyMakeBorder(wordImage[y:y + h, x:x + w], 
                top=(w-h)//2, bottom=-((w-h)//-2), left=None, right=None, 
                borderType=cv.BORDER_CONSTANT, value=[255,255,255])
        else:
            charImage = wordImage[y:y + h, x:x + w]
        
        charImage = cv.resize(charImage, (28,28))

        charImage = np.expand_dims(charImage, axis=0)
        charImage = np.expand_dims(charImage, axis=3)

        string = string + lb.inverse_transform(model.predict(charImage))[0]

    return string


# Loading models
model = load_model(sys.argv[1])
lb = pickle.load(open(sys.argv[2], 'rb'))

image = cv.imread(sys.argv[3])

grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

rowImages = getRows(grayImage)

string = ''
for row in rowImages:
    wordImages = getWords(row)
    for word in wordImages:
        string = string + getChars(word)
        string = string + ' '
    string = string + '\n'

print('\n' + string)