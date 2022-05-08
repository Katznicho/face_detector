import cv2 as cv
from variables import  Variables
from images import Images
MyVariables = Variables(1.1,2)

MyImage =  Images("images/people.jpg")
girl = MyImage.read_image()
MyImage.show_image("Girl Image", MyImage.read_image())
MyImage.convert_to_gray()
MyImage.show_image("Gray Image", MyImage.convert_to_gray())


# call the classifier
haar = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar.detectMultiScale(MyImage.convert_to_gray(), scaleFactor=MyVariables.scale_factor,minNeighbors=MyVariables.get_min_neighbor())

print(f'The number of faces found are {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(girl, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
cv.imshow("Detected Faces", girl)

cv.waitKey(0)
