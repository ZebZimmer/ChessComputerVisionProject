from ultralytics import YOLO
import numpy as np
import cv2
from data_processing_functions import *

class Chess_YOLO:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def load_best_model(self):
        self.model = YOLO("C:/Users/zebzi/Documents/School/Master_Year/CSCI 5525/Project/Models_Saved/YOLOv8_30epochs.pt")

    def train_on_data(self, path_to_data, epoch):
        self.model.train(
            mode='detect',
            data=path_to_data + 'data.yaml',
            imgsz=416,
            epochs=epoch,
            batch=8,
            name=path_to_data + f'{epoch}_epochs_result'
        )

    def board_image_to_piece_locations(self, image: np.array, confidence=0.25):
        '''
        Predict using the YOLOv8 model. The boxes object that is returned holds
        the locations of all predicted bounding boxes. The model has been trained
        to detect chess pieces and bound them in said bounding box.
        '''
        boxes_object = self.model.predict(source=image, conf=confidence)

        num_boxes = len(boxes_object[0].boxes.xywhn)
        image = boxes_object[0].orig_img
        pieces_list = []

        # Corners of the bounding box
        x1, x2, y1, y2 = 0, 0, 0, 0

        for i in range(num_boxes):
            bbox = boxes_object[0].boxes[i].xywhn
            img_h, img_w = boxes_object[0].orig_shape

            x_center, y_center, width, height = bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]

            # Calculate the locations of the top left and bottom right corners
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)

            # cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

            temp_image = np.copy(image)
            cv2.rectangle(temp_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

            cropped_image = image[y1:y2, x1:x2, :]
            padded_image = pad_cropped_image(cropped_image)

            pieces_list.append([padded_image, [x1, y1, x2, y2]])

        return pieces_list

