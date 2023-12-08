import numpy as np
import cv2

def take_photo(index = 1, resize=True):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    ret, frame = cap.read()

    if ret:
        image = np.array(frame)
        if resize:
            image = cv2.resize(image, (416, 416))

    cap.release()
    return image
