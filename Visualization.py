import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from HoughTransform import *


def draw_point_stack(img, point_stack):
    new_img = img.copy()
    h, w = img.shape[:2]

    gradients = []
    line_info = []

    for i in point_stack:
        coeff = hough_intersect(i[0], i[1], img)
        if (len(coeff) > 1):
            m = coeff[0]
            b = coeff[1]
            y1 = m * 100 + b
            y2 = m * (w - 100) + b

            gradients.append(m)
            line_info.append((m, b))
            cv.line(new_img, (100, int(y1)), (w - 100, int(y2)), (255,0,0))


    return new_img, gradients, line_info
