import cv2
import cv2.aruco as aruco
import numpy as np
from photo_functions import *
from collections import Counter # Use this for max checking of board list
from stockfish_eval import send_FEN_get_move

class ChessBoardObj:
    def __init__(self):
        self.board_state = [[[] for i in range(8)] for i in range(8)] # 8x8 'array' where the element will be a list so 8x8xY were Y will grow as we predict pieces in locations
        self.pieces = ["Black Bishop", "Black King", "Black Knight", "Black Pawn", "Black Queen", "Black Rook", "White Bishop", "White King", "White Knight", "White Pawn", "White Queen", "White Rook"]
        self.FEN_pieces = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"] # Pieces in FEN encoding
        self.image = None
        self.labeled_image = None
        self.corners_dict = None
        self.vh_xy_12 = None
        self.guess_list = []

    def detect_aruco_markers(self):
        grey_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()

        corners, ids, _ = aruco.detectMarkers(grey_image, aruco_dict, parameters=parameters)   

        return (corners, ids)                           

    def detect_chessboard(self, crop=False):
        '''
        Take photo of the chess board and use the ArUco markers to find the relative positions of the chess squares.
        Return the 4 corners of the board and the image
        '''
        # Reset everything except for the board_state
        self.image = None
        self.labeled_image = None
        self.corners_dict = None
        self.vh_xy_12 = None
        self.guess_list = []

        good_photo = False
        while not good_photo:
            print("Taking a photo of the board")
            self.image = take_photo(3, resize=(not crop))
            self.labeled_image = self.image
            corners, ids = self.detect_aruco_markers()
            try:
                print(f"Found {ids.shape[0]} aruco tags {[i for i in ids] = }")

                #Crop the image to be just the board
                if crop and ids.shape[0] >= 4:
                    self.corners_dict = {ids[0][0]: (int(corners[0][0][0][0]), int(corners[0][0][0][1])), ids[1][0]: (int(corners[1][0][0][0]), int(corners[1][0][0][1])),
                                        ids[2][0]: (int(corners[2][0][0][0]), int(corners[2][0][0][1])), ids[3][0]: (int(corners[3][0][0][0]), int(corners[3][0][0][1]))}
                    self.image = self.image[self.corners_dict[10][1] - 60 : self.corners_dict[11][1] + 40, self.corners_dict[11][0] - 40 : self.corners_dict[12][0] + 40,  :]

                    self.image = cv2.resize(self.image, (416,416))
                    corners, ids = self.detect_aruco_markers()
                    print(f"#Found {ids.shape[0]} aruco tags {[i for i in ids] = }")
                    self.labeled_image = self.image

                if ids.shape[0] >= 4:
                    good_photo = True

            except:
                print("Bad photo please retake") 

        self.corners_dict = {ids[0][0]: (int(corners[0][0][0][0]), int(corners[0][0][0][1])), ids[1][0]: (int(corners[1][0][0][0]), int(corners[1][0][0][1])),
                        ids[2][0]: (int(corners[2][0][0][0]), int(corners[2][0][0][1])), ids[3][0]: (int(corners[3][0][0][0]), int(corners[3][0][0][1]))}

        return self.image

    def set_x_y_square_offset(self):
        # v for vertial lines, h for horizonal lines
        v_x1 = (self.corners_dict[13][0] - self.corners_dict[10][0]) // 8
        v_y1 = (self.corners_dict[13][1] - self.corners_dict[10][1]) // 8
        v_x2 = (self.corners_dict[12][0] - self.corners_dict[11][0]) // 8
        v_y2 = (self.corners_dict[12][1] - self.corners_dict[11][1]) // 8

        h_x1 = (self.corners_dict[11][0] - self.corners_dict[10][0]) // 8
        h_y1 = (self.corners_dict[11][1] - self.corners_dict[10][1]) // 8
        h_x2 = (self.corners_dict[12][0] - self.corners_dict[13][0]) // 8
        h_y2 = (self.corners_dict[12][1] - self.corners_dict[13][1]) // 8

        self.vh_xy_12 = (v_x1,v_y1,v_x2,v_y2,h_x1,h_y1,h_x2,h_y2)
    
    def draw_square_lines(self, best):
        self.set_x_y_square_offset()
        v_x1,v_y1,v_x2,v_y2,h_x1,h_y1,h_x2,h_y2 = self.vh_xy_12
        offset_list = [100, 10, 16, 18, 22, 22, 14, 8, -1]

        if best:
            cv2.line(self.labeled_image, (self.corners_dict[10][0], self.corners_dict[10][1]), (self.corners_dict[11][0], self.corners_dict[11][1]), (230,0,0,), thickness=3)
            cv2.line(self.labeled_image, (self.corners_dict[10][0], self.corners_dict[10][1]), (self.corners_dict[13][0], self.corners_dict[13][1]), (230,0,0,), thickness=3)
            for i in range(1, 9):
                offset = offset_list[i]
                cv2.line(self.labeled_image, (self.corners_dict[10][0]+(v_x1 * i), self.corners_dict[10][1]+(v_y1 * i)), (self.corners_dict[11][0]+(v_x2 * i), self.corners_dict[11][1]+(v_y2 * i)), (230,0,0,), thickness=3)
                cv2.line(self.labeled_image, (self.corners_dict[10][0]+(h_x1 * i), self.corners_dict[10][1]+(h_y1 * i) - offset), (self.corners_dict[13][0]+(h_x2 * i), self.corners_dict[13][1]+(h_y2 * i) - offset), (230,0,0,), thickness=3)
        else:
            for i in range(8):
                x_upper = self.corners_dict[10][0] + v_x1 * (i + 1)
                x_lower = self.corners_dict[10][0] + v_x1 * i

                offset = h_y1 // 4
                for j in range(8):
                    y_upper = self.corners_dict[10][1] + (h_y1 * (j + 1)) - offset
                    y_lower = self.corners_dict[10][1] + (h_y1 * j) - offset
                    cv2.line(self.labeled_image, (x_lower, y_lower), (x_lower, y_upper), (255,0,0), thickness=3)
                    cv2.line(self.labeled_image, (x_lower, y_lower), (x_upper, y_lower), (255,0,0), thickness=3)
                    cv2.line(self.labeled_image, (x_upper, y_lower), (x_upper, y_upper), (255,0,0), thickness=3)
                    cv2.line(self.labeled_image, (x_lower, y_upper), (x_upper, y_upper), (255,0,0), thickness=3)
                    offset += 2

    def draw_bbox_and_label(self):
        '''
        Get the image with the postion of the bounding box corners
        as [x1, y1, x2, y2] and with the label. Draw the box and add
        the label to the image for readability
        '''
        for position, guess in self.guess_list:
            label = self.pieces[guess]
            # Draw the bounding box
            x1, y1, x2, y2 = position[0], position[1], position[2], position[3]
            cv2.rectangle(self.labeled_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

            # Write the label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.labeled_image, label, (x1, y1 - 5), font, 0.3, (0, 0, 255), 1)

    def process_guess(self, guess, bbox_data):
        self.set_x_y_square_offset()
        v_x1,v_y1,v_x2,v_y2,h_x1,h_y1,h_x2,h_y2 = self.vh_xy_12

        # Location will be the center of the bottom of the bounding box
        location = ((bbox_data[2] + bbox_data[0])/2, bbox_data[3])

        if location[0] < self.corners_dict[11][0] or location[0] > self.corners_dict[12][0] or location[1] < self.corners_dict[10][1] or location[1] > self.corners_dict[11][1]:
            print("Rejected a guess")
            return
        
        self.guess_list.append((bbox_data, guess))

        offset_list = [0, 12, 16, 20, 24, 20, 14, 8, -100]
        for i in range(8):
            v1 = (self.corners_dict[10][0] + (i * v_x1), self.corners_dict[10][1])
            v2 = (self.corners_dict[10][0] + ((i+1) * v_x1), self.corners_dict[10][1])
            v3 = (self.corners_dict[11][0] + ((i+1) * v_x2), self.corners_dict[11][1])
            v4 = (self.corners_dict[11][0] + (i * v_x2), self.corners_dict[11][1])

            if self.point_in_quadrilateral([location[0], location[1]], v1, v2, v3, v4):
                for j in range(8):
                    if (location[1] > (self.corners_dict[10][1] + (h_y1 * j) - offset_list[j])) and (location[1] < (self.corners_dict[10][1] + (h_y1 * (j + 1)) - offset_list[j+1])):
                        self.board_state[j][i].append(self.FEN_pieces[guess])
                        return 0
        
        print(f"Piece did not make it with {bbox_data = }")

    def draw_squares(self):
        '''
        Draw the squares onto the board using the barycentric functions
        Use this to check and see what the spaces are for each square
        '''

        self.set_x_y_square_offset()
        v_x1,v_y1,v_x2,v_y2,h_x1,h_y1,h_x2,h_y2 = self.vh_xy_12

        offset_list = [0, 12, 16, 20, 24, 20, 14, 8, -100]

        # Draw the sections:
        for x in range(416):
            for y in range(416):
                for i in range(8):
                    v1 = (self.corners_dict[10][0] + (i * v_x1), self.corners_dict[10][1] + (v_y1 * i))
                    v2 = (self.corners_dict[10][0] + ((i+1) * v_x1), self.corners_dict[10][1] + (v_y1 * (i+1)))
                    v3 = (self.corners_dict[11][0] + ((i+1) * v_x2), self.corners_dict[11][1] + (v_y2 * i))
                    v4 = (self.corners_dict[11][0] + (i * v_x2), self.corners_dict[11][1] + (v_y2 * (i+1)))

                    if self.point_in_quadrilateral([x, y], v1, v2, v3, v4):
                        for j in range(8):
                            if (y > (self.corners_dict[10][1] + (h_y1 * j) - offset_list[j])) and (y < (self.corners_dict[10][1] + (h_y1 * (j + 1)) - offset_list[j+1])):
                                self.labeled_image[y][x] = [32*i, 0, 32*j]
    
    def print_board(self):
        for i in range(8):
            print("[", end="")
            for j in range(8):
                print(f"{Counter(self.board_state[i][j]).most_common(1)[0][0] if len(self.board_state[i][j]) > 0 else '_'}{',' if j < 7 else ''} ", end="")
            print("]")

    def get_labeled_image(self):
        return self.labeled_image
    
    def sign(self, p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    # Check if a point is in a row by checking if it's on the same side of each line that makes up the box
    def point_in_quadrilateral(self, pt, v1, v2, v3, v4):
        d1 = self.sign(pt, v1, v2)
        d2 = self.sign(pt, v2, v3)
        d3 = self.sign(pt, v3, v4)
        d4 = self.sign(pt, v4, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0) or (d4 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0) or (d4 > 0)

        return not (has_neg and has_pos) 
    
    def get_next_best_move(self):
        
        FEN_str = ""
        for i in range(8):
            count = 0
            for j in range(8):
                if len(self.board_state[i][j]) != 0:
                    if count != 0:
                        FEN_str += f"{count}"
                        count = 0
                    # Single guess path
                    # FEN_str += self.board_state[i][j][0]
                    # List of guesses
                    FEN_str += Counter(self.board_state[i][j]).most_common(1)[0][0]
                else:
                    count += 1

            if count != 0:
                    FEN_str += f"{count}"
            
            FEN_str += "/" if i != 7 else " "
        
        FEN_str += "w - - 0 20"
        print(FEN_str)
        print(send_FEN_get_move(FEN_str))
        


