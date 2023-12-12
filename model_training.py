from CNN_with_CSV_Data import ChessCNN_CSV
from CNN_with_YOLO_BBoxes import ChessCNN_YOLO
import tensorflow as tf
import os
from data_processing_functions import *

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
    print(x_test.shape)
    print(y_test.shape)

    # Create the model using the dimensions of the first image (hopefully they're all the same?)
    model = ChessCNN_YOLO(x_train[0].shape[0], x_train[0].shape[1])

    # Train the model
    model.train(x_train, y_train, x_valid, y_valid, 10)
    
    # Test the model
    model.test(x_test, y_test)
        

def main():
    # train_and_test_CNN_with_CSV_Data()
    train_and_test_CNN_with_YOLO_BBoxes()

    
if __name__ == "__main__":
    print(f"{tf.config.list_physical_devices('GPU')}")
    # Check if TensorFlow is using the GPU
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available and TensorFlow is using it.")
    else:
        print("No GPU is available, TensorFlow is using the CPU.")
    main()