import cv2 as cv
from matplotlib import pyplot as plt

print("Visualization.py loaded")


def PLTdraw(img, color=cv.COLOR_BGR2RGB, binary=False):
    if (not binary):
        plt.figure(figsize=[20, 20])
        return plt.imshow(cv.cvtColor(img, color))
    else:
        plt.figure(figsize=[20, 20])
        return plt.imshow(img)


def draw_ellipses(orgImg, ellipes, color=(0, 234, 255), thickness=2, lst=True):
    img = orgImg.copy()
    if lst:
        for ellipe in ellipes:
            img = cv.ellipse(img, ellipe, color=color, thickness=thickness)
    else:
        img = cv.ellipse(img, ellipes, color=color, thickness=thickness)
    return img


def draw_boxs(orgImg, boxs, color=(255, 0, 0), thickness=2, lst=True):
    img = orgImg.copy()
    if lst:
        for box in boxs:
            ((box_x, box_y), (box_width, box_hight)) = box
            img = cv.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_hight), color=color,
                               thickness=thickness)
    else:
        ((box_x, box_y), (box_width, box_hight)) = boxs
        img = cv.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_hight), color=color,
                           thickness=thickness)
    return img


def draw_pair(orgImg, box, ellipes):
    return draw_boxs(draw_ellipses(orgImg.copy(), ellipes), box, lst=False)
