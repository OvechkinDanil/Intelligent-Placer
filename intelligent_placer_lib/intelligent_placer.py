from typing import Tuple, Optional
import numpy as np
from skimage.morphology import binary_closing
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
import cv2


def find_paper_and_polygon_contours(path_to_img: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    img = cv2.imread(path_to_img)
    contours = _find_contours(img)

    good_contours = []
    cx = []
    cy = []
    polygon_contours = None

    min_contour_area = 10_000
    prox_measure = 14
    max_area = 0
    max_area_idx = -1
    eps = 1e-5

    for i, cnt in enumerate(contours):
        contour_area = cv2.contourArea(cnt)
        if contour_area > max_area:
            max_area = contour_area
            max_area_idx = i

        if contour_area > min_contour_area:
            good_contours.append(cnt)
            moments = cv2.moments(cnt)
            cx.append(int(moments['m10'] / (moments['m00'] + eps)))
            cy.append(int(moments['m01'] / (moments['m00'] + eps)))

    for i in range(len(good_contours) - 1):
        if np.linalg.norm([cx[i] - cx[i + 1], cy[i] - cy[i + 1]]) < prox_measure:
            polygon_contours = good_contours[i]
            good_contours.pop(i + 1)
            good_contours.pop(i)

    paper_contours = good_contours[max_area_idx]

    cv2.drawContours(img, [paper_contours], 0, (255, 0, 0), 8)
    cv2.drawContours(img, [polygon_contours], 0, (0, 255, 0), 8)

    return polygon_contours, paper_contours


def _find_contours(img: np.ndarray):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    in_range_checker = cv2.inRange(img_hsv, np.array([0, 0, 169]), np.array([255, 255, 255]))
    st = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))
    morph_close_transform = cv2.morphologyEx(in_range_checker, cv2.MORPH_CLOSE, st)
    st = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50), (-1, -1))
    morph_open_transform = cv2.morphologyEx(morph_close_transform, cv2.MORPH_OPEN, st)

    contours, _ = cv2.findContours(morph_open_transform, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def get_object_rectangles(path_to_img: str, paper_contours: Optional[np.ndarray]):
    min_paper_y = min(paper_contours, key=lambda coordinates: coordinates[0][1])[0][1]

    image = cv2.imread(path_to_img)[:min_paper_y, :]
    (h, w, d) = image.shape

    contours = find_all_contours(image)

    objects = list(contours[0])
    cv2.drawContours(image, objects, -1, (0, 255, 0), thickness=4)

    rectangles = find_rectangles_for_objects(objects, h, w)

    cv2.drawContours(image, rectangles, -1, (0, 0, 255), 4)

    cv2.imshow('Binary image', image)
    cv2.waitKey(0)
    cv2.imwrite('../src/image_thres1.jpg', image)
    cv2.destroyAllWindows()

    return rectangles


def find_all_contours(image: np.ndarray):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge = 255 * binary_fill_holes(binary_closing(canny(gray_img, sigma=1), selem=np.ones((5, 5))))
    edge = edge.astype(np.uint8)

    return cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


def find_rectangles_for_objects(objects: list, h, w):
    rectangles = list()
    for obj in objects:
        rect = cv2.minAreaRect(obj)
        rectangle = np.int0(cv2.boxPoints(rect))
        if is_rectangle_valid(rectangle, h, w):
            rectangles.append(rectangle)

    return rectangles


def is_rectangle_valid(rectangle, h, w):
    delta = 50

    cond = lambda elem: elem[0] < delta or elem[1] < delta or elem[0] >= w - delta or elem[1] >= h - delta
    if sum(cond(elem) for elem in rectangle) == len(rectangle):
        return False

    min_allowable_area = 550
    max_allowable_area = (h - delta) * (w - delta)

    return max_allowable_area > cv2.contourArea(rectangle) > min_allowable_area


def check_image(path_to_image: str):
    polygon_contours, paper_contours = find_paper_and_polygon_contours(path_to_image)
    object_rectangles = get_object_rectangles(path, paper_contours)
    sum_areas = sum(cv2.contourArea(obj) for obj in object_rectangles)

    return cv2.contourArea(polygon_contours) >= sum_areas


if __name__ == '__main__':
    path = "D:/study/Intelligent-Placer/data/yes/test7.jpg"
    result = check_image(path)
    print(result)
