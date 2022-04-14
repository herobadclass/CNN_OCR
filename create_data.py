# python create_data.py ascii.png

import sys
import string
import cv2 as cv
import numpy as np
import pandas as pd

ascii = cv.imread(sys.argv[1])

grayImage = cv.cvtColor(ascii, cv.COLOR_BGR2GRAY)

bitImage = ~cv.threshold(grayImage, 0, 255, cv.THRESH_OTSU)[1]

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
    rowImages.append(grayImage[y:y + h, x:x + w])

images = []
labels = []
characters = list(string.ascii_letters + string.digits + string.punctuation)
i = 0

# Extracting ascii characters
for image in rowImages:
    bitImage = ~cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,3,22)
    
    # Dilating text vertically
    kernel = np.array(
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]], np.uint8)
    dilatedImage = cv.dilate(bitImage, kernel, iterations=2)

    # Getting row coordinates and sorting by X axis
    contours = cv.findContours(dilatedImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    chars = []
    for contour in contours:
        chars.append(cv.boundingRect(contour))
    chars = sorted(chars, key=lambda x: x[0])

    # Adding margin to create a square image and resizing
    for char in chars:
        x, y, w, h = char
        if h > w:
            charImage = cv.copyMakeBorder(image[y:y + h, x:x + w], 
            top=None, bottom=None, left=(h-w)//2, right=-((h-w)//-2), 
            borderType=cv.BORDER_CONSTANT, value=[255,255,255])
        elif w > h:
            charImage = cv.copyMakeBorder(image[y:y + h, x:x + w], 
            top=(w-h)//2, bottom=-((w-h)//-2), left=None, right=None, 
            borderType=cv.BORDER_CONSTANT, value=[255,255,255])
        else:
            charImage = image[y:y + h, x:x + w]
        charImage = cv.resize(charImage, (28,28))

        path = 'ascii/' + str(i) + '.png'
        
        labels.append(characters[i])
        images.append(path) 

        cv.imwrite(path, charImage)
        
        i += 1

df = pd.DataFrame(list(zip(images, labels)), columns=['images', 'labels'])
df.to_csv('data.csv')