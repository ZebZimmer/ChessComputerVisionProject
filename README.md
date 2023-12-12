# Live Chessboard Analysis Using Computer Vision

## Description
This project aims to automate the process of analyzing a live chessboard using computer vision techniques. By capturing images of a chessboard during gameplay, the system detects chess pieces, identifies their positions, and suggests the next best move by interfacing with a modern chess engine.

## Key Features
- **Chess Piece Detection**: Utilizes a YOLOv8 model for detecting chess pieces on a standard 64-square chessboard.
- **Piece Classification**: Employs a Convolutional Neural Network (CNN) for classifying the detected chess pieces.
- **Chessboard Analysis**: Analyzes the board state and integrates with the Stockfish API to recommend the next best move.
- **Data Preprocessing**: Implements specialized techniques for image preprocessing to optimize model performance.

## Technical Overview
- **Models Used**: 
  - YOLOv8 Model: For detecting chess pieces with bounding box coordinates.
  - CNN Model: For classifying the detected pieces into one of the 12 distinct chess piece types.
- **Data and Preprocessing**: The training data is sourced from Roboflow, with images preprocessed for consistency and to reduce distortion.
- **Model Performance**: Enhanced by minor technique changes, including taking multiple images and slight camera perturbations for varied data.

## Installation and Usage
- The replication of this project requires setting up ArUco markers on the corners of the board.
  - From the White player's perspective the IDs go 10, 11, 12, 13 in counter-clockwise order starting in the top left corner (a8).
  An example can be seen in the image below.
- After setting up the board
  - Create a python virtual environment inside of the project folder: `python -m venv /ChessEnv`  
  - Activate the virtual environment  
      -Windows: `.\ChessEnv\Scripts\Activate`  
      -Mac: `source ChessEnv\bin\activate`  
  - Install the necessary libraries with `pip install -r requirement.txt`  
  - Run the live_model_implementation.py script in this environment and see the results live

## Results
- Overview of the model training, performance metrics, and real-world application results.
- Visual examples demonstrating the system's ability to analyze chessboard states and predict moves.
  ![Example of Board Analysis](https://github.com/ZebZimmer/ChessComputerVisionProject/blob/main/Pictures/perfect_read.png)  
On C5 the knight is misclassified as a rook. The postprocessing technique, however, of averaging the predictions over 10 images made the final board analysis be 100% correct with a knight in C5. Stockfish returned that the best move in this position for White is C2A2 which is the first move in a two-move checkmate for white.

## Future Work
- Developing a user-friendly GUI to simplify interactions with the system.
- Potential enhancements to model accuracy and generalizability.
