from typing import Tuple, Optional

import numpy as np
import cv2


def find_paper_and_polygon_contours(path_to_img: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    img = cv2.imread(path_to_img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    filter = cv2.inRange(img_hsv, np.array([0, 0, 169]), np.array([255, 255, 255]))
    st = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))
    filter = cv2.morphologyEx(filter, cv2.MORPH_CLOSE, st)
    st = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50), (-1, -1))
    filter = cv2.morphologyEx(filter, cv2.MORPH_OPEN, st)

    contours, hierarchy = cv2.findContours(filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    good_contours = []
    cx = []
    cy = []
    polygon_contours = None
    area_contours = []

    min_contour_area = 10_000
    prox_measure = 14

    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        if contour_area > min_contour_area:
            area_contours.append(contour_area)
            good_contours.append(cnt)
            M = cv2.moments(cnt)
            cx.append(int(M['m10'] / (M['m00'] + 1e-5)))
            cy.append(int(M['m01'] / (M['m00'] + 1e-5)))

    for i in range(len(good_contours) - 1):
        if np.linalg.norm([cx[i] - cx[i + 1], cy[i] - cy[i + 1]]) < prox_measure:
            polygon_contours = good_contours[i]
            good_contours.pop(i + 1)
            good_contours.pop(i)

    idx = area_contours.index(max(area_contours))
    paper_contours = good_contours[idx]

    return polygon_contours, paper_contours
