import cv2
import cv2.aruco as aruco
import numpy as np
from live_model_implementation import *

def detect_aruco_markers(image):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    # if ids is not None:
    #     aruco.drawDetectedMarkers(image, corners, ids)
    #     cv2.imshow('Detected ArUco markers', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()  
    #     print(f"{corners = }")
    #     print(f"{ids = }")

    # else:
    #     print("No marker detected :(")     

    return (corners, ids)                           

def detect_chessboard(image, draw_lines=False):
    '''
    Take photo of the chess board and use the ArUco markers to find the relative positions of the chess squares.
    Return the 4 corners of the board
    '''
    ids = np.array((3,3))
    while ids.shape[0] < 4:
        print("Taking a photo of the board")
        image = take_photo(3)
        corners, ids = detect_aruco_markers(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    corners_dict = {ids[0][0]: (int(corners[0][0][0][0]), int(corners[0][0][0][1])), ids[1][0]: (int(corners[1][0][0][0]), int(corners[1][0][0][1])),
                    ids[2][0]: (int(corners[2][0][0][0]), int(corners[2][0][0][1])), ids[3][0]: (int(corners[3][0][0][0]), int(corners[3][0][0][1]))}

    # v for vertial, h for horizonal
    v_x1 = (corners_dict[13][0] - corners_dict[10][0]) // 8
    v_y1 = (corners_dict[13][1] - corners_dict[10][1]) // 8
    v_x2 = (corners_dict[12][0] - corners_dict[11][0]) // 8
    v_y2 = (corners_dict[12][1] - corners_dict[11][1]) // 8

    h_x1 = (corners_dict[11][0] - corners_dict[10][0]) // 8
    h_y1 = (corners_dict[11][1] - corners_dict[10][1]) // 8
    h_x2 = (corners_dict[12][0] - corners_dict[13][0]) // 8
    h_y2 = (corners_dict[12][1] - corners_dict[13][1]) // 8

    offset = h_y1 // 4
    cv2.line(image, (corners_dict[10][0], corners_dict[10][1]), (corners_dict[11][0], corners_dict[11][1]), (230,0,0,), thickness=3)
    cv2.line(image, (corners_dict[10][0], corners_dict[10][1]), (corners_dict[13][0], corners_dict[13][1]), (230,0,0,), thickness=3)
    for i in range(1, 9):
        cv2.line(image, (corners_dict[10][0]+(v_x1 * i), corners_dict[10][1]+(v_y1 * i)), (corners_dict[11][0]+(v_x2 * i), corners_dict[11][1]+(v_y2 * i)), (230,0,0,), thickness=3)
        cv2.line(image, (corners_dict[10][0]+(h_x1 * i), corners_dict[10][1]+(h_y1 * i) - offset), (corners_dict[13][0]+(h_x2 * i), corners_dict[13][1]+(h_y2 * i) - offset), (230,0,0,), thickness=3)
        offset += 2

    cv2.imshow('line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  


def main():
    pass


if __name__ == "__main__":
    # This is just to debug the function(s)
    main()
