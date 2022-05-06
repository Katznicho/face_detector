import cv2 as cv
import numpy as np

girl_image = cv.imread("images/people.jpg")
cv.imshow("Girl Image", girl_image)

# convert image to gray scale
gray = cv.cvtColor(girl_image, cv.COLOR_BGR2GRAY);
cv.imshow("Gray Scale ", gray)
# call the classifier
haar = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

#print(f'The number of faces found are {faces_rect}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(girl_image, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
cv.imshow("Detected Faces", girl_image)

cv.waitKey(0)
