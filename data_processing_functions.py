from PIL import Image
import numpy as np
import csv
from tqdm import tqdm
import os
import cv2

datasets_filepath_CSV = r'C:\Users\zebzi\Documents\School\Master_Year\CSCI 5525\Project\CSV_Multiclass'
datasets_filepath_YOLO = r'C:\Users\zebzi\Documents\School\Master_Year\CSCI 5525\Project\YOLOv8'

def get_CSV_data_return_numpy_array(type: str) -> (np.array, np.array):
    """
    Get the type (train, test, valid) of the data and return it as two arrays
    The first is an array of arrays where each index holds data for an image
    The second array holds the data labels
    """
    # labels_dict = {}
    photos_array = []
    labels_array = []

    with open(datasets_filepath_CSV + f"\{type}\_classes.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in tqdm(csvreader):
            if(row[0] != "filename"):
                # labels_dict[row[0]] = np.array(row[2:], int)
                image = Image.open(datasets_filepath_CSV + f"\{type}\\" + row[0])
                photos_array.append(np.array(image.resize((350, 350))))
                labels_array.append(np.array(row[2:], int))

    return (np.array(photos_array), np.array(labels_array))

def get_YOLO_data_return_numpy_array(type: str) -> (np.array, np.array):
    """
    Get the type (train, test, valid) of the data and return it as two arrays
    The images will have multiple pieces but this function will create an entry
    to the first array with simply the piece (only the contents of a single 
    bounding box). The second array will hold the label for the corresponding image.
    """
    photos_array = []
    labels_array = []

    for _, _, files in os.walk(datasets_filepath_YOLO + f"/{type}_data/images/"):
        for file in files:
            full_image = np.array(Image.open(datasets_filepath_YOLO + f"/{type}_data/images/{file}"))
            img_h = full_image.shape[0]
            img_w = full_image.shape[1]
            with open(datasets_filepath_YOLO + f"/{type}_data/labels/{file[:-4]}.txt") as label_file:
                for data in label_file:
                    data = data.split(" ")

                    # Get the bounding box data from the label
                    label, x, y, w, h = float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])

                    # Bounding box values are relative to the img size so get the actual 'pixels'
                    x1 = int((x - w / 2) * img_w)
                    y1 = int((y - h / 2) * img_h)
                    x2 = int((x + w / 2) * img_w)
                    y2 = int((y + h / 2) * img_h)

                    # Crop the image to the bounding box
                    cropped_image = full_image[y1:y2, x1:x2]

                    padded_image = pad_cropped_image(cropped_image)

                    # Append the image and the label to the list
                    photos_array.append(cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY))
                    if label > 5:
                        label = label - 6

                    blank_label_array = [0 for i in range(6)]
                    blank_label_array[int(label) - 1] = 1
                    labels_array.append(blank_label_array)
    
    return (np.array(photos_array), np.array(labels_array))

def pad_cropped_image(cropped_image):
    cropped_image_size = [101, 46, 1] # This is the largest cropped image size from the entire dataset
    # Pad the image to the largest cropped image size (calculated by 'hand')
    pad_h = max(cropped_image_size[0] - cropped_image.shape[0], 0)
    pad_w = max(cropped_image_size[1] - cropped_image.shape[1], 0)

    pad_h = (pad_h // 2, pad_h - (pad_h // 2))
    pad_w = (pad_w // 2, pad_w - (pad_w // 2))

    padded_image = np.pad(cropped_image, ((pad_h), (pad_w), (0,0)))

    return padded_image