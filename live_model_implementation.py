from YOLOv8_model import Chess_YOLO
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

pieces = ["Black Bishop", "Black King", "Black Knight", "Black Pawn", "Black Queen", "Black Rook", "White Bishop", "White King", "White Knight", "White Pawn", "White Queen", "White Rook"]

def take_photo(dshow = False):
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    ret, frame = cap.read()

    if ret:
        image = np.array(frame)
        image = cv2.resize(image, (416, 416))

    cap.release()
    return image

def draw_bbox_and_label(image, position, label):
    '''
    Get the image with the postion of the bounding box corners
    as [x1, y1, x2, y2] and with the label. Draw the box and add
    the label to the image for readability
    '''
    # Draw the bounding box
    x1, y1, x2, y2 = position[0], position[1], position[2], position[3]
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

    # Write the label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, label, (x1, y1 - 5), font, 0.3, (0, 0, 255), 1)


def predict_from_board_photo(CNN_model, YOLO_model, image: np.array):
    '''
    Take in a photo of the board and return the position and label
    for every piece on the board as [[label, [x, y]], ...]
    The image should be a numpy array and it will be drawn on 
    '''
    image_with_bbox_data_list = YOLO_model.board_image_to_piece_locations(image)
    labelled_image = np.copy(image)

    # Cropped image for the CNN to decipher the piece and the bbox data so that
    # the original image can be edited 
    for cropped_image, bbox_data in image_with_bbox_data_list:
        try:
            CNN_percents = CNN_model(np.reshape(cropped_image, (1, 101, 46, 3)))
        except:
            pass
        guess = np.argmax(CNN_percents.numpy())

        draw_bbox_and_label(labelled_image, bbox_data, pieces[guess])
        # print(pieces[guess])
        # print(CNN_percents.numpy())
        # print(bbox_data)
        # print()

    cv2.imshow("Picture with labels", labelled_image)

def main():
    YOLO_model = Chess_YOLO()
    YOLO_model.load_best_model()
    
    while(True):
        CNN_with_YOLO_model = load_model("C:/Users/zebzi/Documents/School/Master_Year/CSCI 5525/Project/Models_Saved/CNN_with_YOLO_BBoxes_30epochs.keras")
        # full_image = np.array(Image.open("C:/Users/zebzi/Documents/School/Master_Year/CSCI 5525/Project/YOLOv8/test_data/images/410993714e325a1de3e394ffe860df3a_jpg.rf.657c49ca295ef54da23469189070a075.jpg").resize([416, 416]))
        full_image = take_photo()
        predict_from_board_photo(CNN_with_YOLO_model, YOLO_model, full_image)
        plt.imshow(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB))
        plt.show(block=False)

        plt.pause(1)

        plt.close('all')





    
if __name__ == "__main__":
    print(f"{tf.config.list_physical_devices('GPU')}")
    # Check if TensorFlow is using the GPU
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available and TensorFlow is using it.")
    else:
        print("No GPU is available, TensorFlow is using the CPU.")
    main()