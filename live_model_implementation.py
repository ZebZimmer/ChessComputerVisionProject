from YOLOv8_model import Chess_YOLO
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from ChessBoard import ChessBoardObj

def predict_from_board_photo(CNN_model, YOLO_model, image: np.array, ChessBoard):
    '''
    Take in a photo of the board and return the position and label
    for every piece on the board as [[label, [x, y]], ...]
    The image should be a numpy array and it will be drawn on 
    '''
    image_with_bbox_data_list = YOLO_model.board_image_to_piece_locations(image)
    # labelled_image = np.copy(image)

    # Cropped image for the CNN to decipher the piece and the bbox data so that
    # the original image can be edited 
    for cropped_image, bbox_data in image_with_bbox_data_list:
        try:
            CNN_percents = CNN_model(np.reshape(cropped_image, (1, 101, 46, 3)))
            
            guess = np.argmax(CNN_percents.numpy())
            _ = ChessBoard.process_guess(guess, bbox_data)
        except:
            print("Skipped in piece detection")
            pass
        
        # print(pieces[guess])
        # print(CNN_percents.numpy())
        # print(bbox_data)
        # print()

    

def main():
    
    YOLO_model = Chess_YOLO()
    YOLO_model.load_best_model()

    CNN_with_YOLO_model = load_model("C:/Users/zebzi/Documents/School/Master_Year/CSCI 5525/Project/Models_Saved/CNN_with_YOLO_BBoxes_40epochsT.keras")

    while(True):
        ChessBoard = ChessBoardObj()
        for i in range(10):
            full_image = ChessBoard.detect_chessboard(crop=True)

            # Use the models to predict pieces
            predict_from_board_photo(CNN_with_YOLO_model, YOLO_model, full_image, ChessBoard)

        # Drawing functions
        # ChessBoard.draw_square_lines(best=True)
        ChessBoard.draw_bbox_and_label()
        # ChessBoard.draw_squares()

        ChessBoard.print_board()

        ChessBoard.get_next_best_move()

        cv2.imshow('Output', cv2.resize(ChessBoard.get_labeled_image(), (832, 832)))
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

        # plt.imshow(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB))
        # plt.show(block=False)

        # plt.pause(10)

        # plt.close('all')


if __name__ == "__main__":
    print(f"{tf.config.list_physical_devices('GPU')}")
    # Check if TensorFlow is using the GPU
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available and TensorFlow is using it.")
    else:
        print("No GPU is available, TensorFlow is using the CPU.")
    main()