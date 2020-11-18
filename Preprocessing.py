import numpy as np
from PIL import Image

def resize(input_im, height):
    """
    Reszue the given image to have the given height
    input_im - PIL Image object
    height - desired height
    """
    width = int(input_im.width * height / input_im.height)
    resized = input_im.resize((width , height))
    return np.array(resized)

def rgb_to_gray(rgb):
    """
    Covert RGB image in to gray
    rgb - 3-D numpy array.
    """
    grayed = 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]
    return grayed

# For each non-zero pixel in mask, the corresponding pixel on image is kept (the rest of the pixels in mask is set to 0)
def apply_mask(image, mask):
    """
    For each elements in image, set its value to 0 if the matching element in mask is 0.
    image - N-D numpy array.
    mask - N-D numpy array of the same shape.
    """
    return np.ma.array(image,mask=mask == 0).filled(fill_value=0)


def fill_mask(image):
    """
    Create tranigular ROI mask.
    mask - starting mask.
    """
    mask = np.zeros_like(image)
    h, w = mask.shape
    bottom_left = (h, 0)
    middle = (int(h/2), int(w/2))
    bottom_right = (h, w)

    for x, row in enumerate(mask):
        for y, col in enumerate(row):
            # Applying equations to left_bound and right_bound
            left_bound = (h - x) * middle[1] / middle[0]
            right_bound = x * middle[1] / middle[0]
            if y > left_bound and y < right_bound and x <= 400:
                mask[x][y] = 255
    return mask
