import numpy as np
import math


def hough_lines(img, line_length=50):

    height, width = img.shape[:2]
    accumulator = np.zeros((180, int(math.sqrt(height ** 2 + width ** 2))), dtype=np.int)


    lines = np.array([[0, 0], [0, 0]])

    #line_length = 40

    # look for every pixel
    for y in range(0, height):
        for x in range(0, width):
            # if pixel is black (possible part of a line)
            if img[y][x] > 5:
                line = []
                # try all angles
                for theta in range(0, 180):
                    p = int(x * math.cos(math.radians(theta)) + y * math.sin(math.radians(theta)))
                    accumulator[theta][p] += 1
                    # Check if it looks like line and if it's not in a list
                    if (accumulator[theta][p] > line_length) and (p not in lines[:, 0]) and (theta not in  lines[:, 1]):
                        lines = np.vstack((lines, np.array([p, theta])))

    # clean two first zeros
    lines = np.delete(lines, [0, 1], axis=0)

    return accumulator, lines

# Outputs gradient and y intersect of the line for y = mx + b
def hough_intersect(rho, theta, image):
    h, w = image.shape[:2]
    out = []
    line_info = []
    theta = math.radians(theta)
    intersect = [int(round(rho / math.sin(theta))), int(round((rho - w * math.cos(theta)) / math.sin(theta))), int(round(rho / math.cos(theta))),
                 int(round((rho - h * math.sin(theta)) / math.cos(theta)))]


    # Use formula to get 2 points in a line, one being 100 pixels from left and another being 100 pixels from right
    x1 = 100
    y1 = int(x1 * (-math.cos(theta)/math.sin(theta)) + rho / math.sin(theta))

    x2 = 800
    y2 = int(x2 * (-math.cos(theta)/math.sin(theta)) + rho / math.sin(theta))

    # Gradient
    m = -math.cos(theta)/math.sin(theta)
    # Intersect
    b = rho / math.sin(theta)

    out.append(m)
    out.append(b)

    line_info.append(m)
    line_info.append(b)

    return out
