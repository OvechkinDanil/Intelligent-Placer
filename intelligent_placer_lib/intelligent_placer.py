from typing import Tuple, Optional
import numpy as np
from skimage.morphology import binary_closing
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
import cv2
import skimage
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def create_mask_from_contours(contours: np.ndarray) -> np.ndarray:
    bounding_rect = cv2.boundingRect(contours)
    y_down, x_left, height, width = bounding_rect[0], bounding_rect[1], bounding_rect[2], bounding_rect[3]
    mask = np.full((width, height), True)

    for y in range(y_down, y_down + height):
        for x in range(x_left, x_left + width):
            if cv2.pointPolygonTest(contours, (y, x), False) >= 0:
                mask[x - x_left][y - y_down] = False

    return mask


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


def get_objects_masks_with_areas(path_to_img: str, paper_contours: Optional[np.ndarray]):
    min_paper_y = min(paper_contours, key=lambda coordinates: coordinates[0][1])[0][1]

    image = cv2.imread(path_to_img)[:min_paper_y, :]
    (h, w, d) = image.shape

    contours = find_all_contours(image)

    objects = list(contours[0])

    object_masks_with_areas = []
    for obj in objects:
        rect = cv2.minAreaRect(obj)
        rectangle = np.int0(cv2.boxPoints(rect))
        if is_rectangle_valid(rectangle, h, w):
            mask = create_mask_from_contours(obj)
            mask = ~mask
            label = skimage.measure.label(mask)
            prop = skimage.measure.regionprops(label)[0]
            object_masks_with_areas.append((mask, prop.area))

    return object_masks_with_areas


def find_all_contours(image: np.ndarray):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge = 255 * binary_fill_holes(binary_closing(canny(gray_img, sigma=1), selem=np.ones((5, 5))))
    edge = edge.astype(np.uint8)

    return cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


def find_rectangles_for_objects(objects: list, h, w):
    object_masks_with_area = list()
    for obj in objects:
        rect = cv2.minAreaRect(obj)
        rectangle = np.int0(cv2.boxPoints(rect))
        if is_rectangle_valid(rectangle, h, w):
            mask = create_mask_from_contours(obj)
            mask = ~mask
            object_masks_with_area.append((mask, cv2.contourArea(obj)))

    return object_masks_with_area


def is_rectangle_valid(rectangle, h, w):
    delta = 50

    cond = lambda elem: elem[0] < delta or elem[1] < delta or elem[0] >= w - delta or elem[1] >= h - delta
    if sum(cond(elem) for elem in rectangle) == len(rectangle):
        return False

    min_allowable_area = 1000
    max_allowable_area = (h - delta) * (w - delta)

    return max_allowable_area > cv2.contourArea(rectangle) > min_allowable_area


def try_to_fit_object(extended_polygon_mask, object_mask, pos_x, pos_y):
    object_mask_height, object_mask_width = object_mask.shape
    delta = 25
    step_angle = 20
    max_angle = 360

    extended_polygon_mask_height, extended_polygon_mask_width = extended_polygon_mask.shape

    for y in range(pos_y, extended_polygon_mask_height - object_mask_height, delta):
        for x in range(pos_x, extended_polygon_mask_width - object_mask_width, delta):

            for angle in range(0, max_angle, step_angle):

                rotated_object_mask = rotate(object_mask, angle, reshape=True)
                rotated_object_mask_height, rotated_object_mask_width = rotated_object_mask.shape
                polygon_mask_cut = extended_polygon_mask[y:y + rotated_object_mask_height, x:x +
                                                                                             rotated_object_mask_width]

                try:
                    overlay_areas = cv2.bitwise_and(polygon_mask_cut.astype(int), rotated_object_mask.astype(int))
                except:
                    continue

                if np.sum(overlay_areas) == 0:
                    extended_polygon_mask[y:y + rotated_object_mask_height, x:x + rotated_object_mask_width] = \
                        cv2.bitwise_xor(polygon_mask_cut.astype(int), rotated_object_mask.astype(int)).astype(bool)

                    plt.imshow(extended_polygon_mask)
                    plt.show()
                    return True

    return False


def check_image(path_to_image: str):
    polygon_contours, paper_contours = find_paper_and_polygon_contours(path_to_image)
    object_masks_with_areas = get_objects_masks_with_areas(path_to_image, paper_contours)
    polygon_mask = create_mask_from_contours(polygon_contours)

    sorted_object_masks_with_areas = sorted(object_masks_with_areas, key=lambda tup: tup[1], reverse=True)

    extended_polygon_mask = np.ones(np.asarray(polygon_mask.shape) * 2)
    polygon_mask_h, polygon_mask_w = polygon_mask.shape
    start_pos_y, start_pos_x = np.asarray(extended_polygon_mask.shape) // 2 - np.asarray(polygon_mask.shape) // 2
    extended_polygon_mask[start_pos_y:start_pos_y + polygon_mask_h, start_pos_x:start_pos_x + polygon_mask_w] = \
        polygon_mask

    for object_mask, _ in sorted_object_masks_with_areas:
        if not try_to_fit_object(extended_polygon_mask, object_mask, start_pos_x, start_pos_y):
            return False

    return True
