from PIL import Image
import numpy as np
import csv
from CNN import ChessCNN
from tqdm import tqdm
import tensorflow as tf

datasets_filepath = r'C:\Users\zebzi\Documents\School\Master_Year\CSCI 5525\Project\CSV_Multiclass'

def get_data_return_numpy_array(type: str) -> (np.array, np.array):
    """
    Get the type (train, test, valid) of the data and return it as two arrays
    The first is an array of arrays where each index holds data for an image
    The second array holds the data labels
    """
    # labels_dict = {}
    photos_array = []
    labels_array = []

    with open(datasets_filepath + f"\{type}\_classes.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in tqdm(csvreader):
            if(row[0] != "filename"):
                # labels_dict[row[0]] = np.array(row[2:], int)
                image = Image.open(datasets_filepath + f"\{type}\\" + row[0])
                photos_array.append(np.array(image.resize((350, 350))))
                labels_array.append(np.array(row[2:], int))


    # print(labels_dict["c4943d83c06a12ad5e0399d19514a4ca_jpg.rf.8b0040b3b68009f6f700ea28fb1aa491.jpg"])
    return (np.array(photos_array), np.array(labels_array))
        


def main():
    # Get the data
    x_train, y_train = get_data_return_numpy_array("train")
    x_valid, y_valid = get_data_return_numpy_array("valid")
    x_test, y_test   = get_data_return_numpy_array("test")

    print(x_train[1].shape)
    print(x_valid[2].shape)

    # Create the model using the dimensions of the first image (hopefully they're all the same?)
    model = ChessCNN(x_train[0].shape[0], x_train[0].shape[1])

    # Train the model
    model.train(x_train, y_train, x_valid, y_valid, 30)
    
    # Test the model
    model.test(x_test, y_test)
    
if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    # Check if TensorFlow is using the GPU
    if tf.test.is_gpu_available():
        print("GPU is available and TensorFlow is using it.")
    else:
        print("No GPU is available, TensorFlow is using the CPU.")
    main()