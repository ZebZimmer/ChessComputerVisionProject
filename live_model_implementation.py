from YOLOv8_model import Chess_YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from ChessBoard import ChessBoardObj
import matplotlib.pyplot as plt

def predict_from_board_photo(CNN_model, YOLO_model, image: np.array, ChessBoard):
    '''
    Take in a photo of the board and return the position and label
    for every piece on the board as [[label, [x, y]], ...]
    The image should be a numpy array and it will be drawn on 
    '''
    image_with_bbox_data_list = YOLO_model.board_image_to_piece_locations(image)

    # Cropped image for the CNN to decipher the piece and the bbox data so that
    # the original image can be edited 
    for cropped_image, bbox_data in image_with_bbox_data_list:
        try:
            CNN_percents = CNN_model(np.reshape(cropped_image, (1, 101, 46)))

            guess = np.argmax(CNN_percents.numpy())
            if (np.mean(cropped_image[48:52, 20:26]) > 125):
                guess += 6 #White piece

            _ = ChessBoard.process_guess(guess, bbox_data)
        except:
            print("Skipped in piece detection")
            pass

    

def main():
    
    YOLO_model = Chess_YOLO()
    YOLO_model.load_best_model()

    CNN_with_YOLO_model = load_model("C:/Users/zebzi/Documents/School/Master_Year/CSCI 5525/Project/Models_Saved/CNN_with_YOLO_BBoxes_40epochsG.keras")

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


if __name__ == "__main__":
    print(f"{tf.config.list_physical_devices('GPU')}")
    # Check if TensorFlow is using the GPU
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available and TensorFlow is using it.")
    else:
        print("No GPU is available, TensorFlow is using the CPU.")
    main()