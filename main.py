from PIL import Image
import numpy as np
import csv
from CNN_with_CSV_Data import ChessCNN_CSV
from tqdm import tqdm
import tensorflow as tf
import os
import matplotlib.pyplot as plt
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


    # print(labels_dict["c4943d83c06a12ad5e0399d19514a4ca_jpg.rf.8b0040b3b68009f6f700ea28fb1aa491.jpg"])
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
                    data.split(" ")

                    # Get the bounding box data from the label
                    label, x, y, w, h = float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])

                    # Bounding box values are relative to the img size so get the actual 'pixels'
                    x1 = int((x - w / 2) * img_w)
                    y1 = int((y - h / 2) * img_h)
                    x2 = int((x + w / 2) * img_w)
                    y2 = int((y + h / 2) * img_h)

                    # Crop the image to the bounding box
                    cropped_image = full_image[y1:y2, x1:x2]
                    photos_array.append(cropped_image)
                    labels_array.append(label)
                    


def train_and_test_CNN_with_CSV_Data():
    '''
    Train the CNN on the CSV Data. That data is the pieces on the board where the
    label is simply what pieces are present in the image
    '''
    # Get the data
    x_train, y_train = get_CSV_data_return_numpy_array("train")
    x_valid, y_valid = get_CSV_data_return_numpy_array("valid")
    x_test, y_test   = get_CSV_data_return_numpy_array("test")

    print(x_train[1].shape)
    print(x_valid[2].shape)

    # Create the model using the dimensions of the first image (hopefully they're all the same?)
    model = ChessCNN_CSV(x_train[0].shape[0], x_train[0].shape[1])

    # Train the model
    model.train(x_train, y_train, x_valid, y_valid, 30)
    
    # Test the model
    model.test(x_test, y_test)

def train_and_test_CNN_with_YOLO_BBoxes():
    '''
    Train the CNN on images of only a single chess piece, blown up to fill the image.
    '''
    # Get the data
    x_train, y_train = get_YOLO_data_return_numpy_array("train")
    x_valid, y_valid = get_YOLO_data_return_numpy_array("valid")
    x_test, y_test   = get_YOLO_data_return_numpy_array("test")

    # Create the model using the dimensions of the first image (hopefully they're all the same?)
    model = ChessCNN_CSV(x_train[0].shape[0], x_train[0].shape[1])

    # Train the model
    model.train(x_train, y_train, x_valid, y_valid, 30)
    
    # Test the model
    model.test(x_test, y_test)
        

def main():
    # train_and_test_CNN_with_CSV_Data()
    train_and_test_CNN_with_YOLO_BBoxes()

    # Nick said to explain in detail why the Standard CNN probably failed.
    # The main reason being that there are only 293 unique images. 

    
if __name__ == "__main__":
    print(f"{tf.config.list_physical_devices('GPU')}")
    # Check if TensorFlow is using the GPU
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available and TensorFlow is using it.")
    else:
        print("No GPU is available, TensorFlow is using the CPU.")
    main()