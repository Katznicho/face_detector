import cv2 as cv


class Images:

    def __init__(self, path):
        self.path = path

    def read_image(self):
        return cv.imread(self.path)

    def convert_to_gray(self):
        return cv.cvtColor(self.read_image(), cv.COLOR_BGR2GRAY)

    def show_image(self, text, image):
        return cv.imshow(text, image)

    def use_wait_key(self, delay):
        return cv.waitKey(delay)

