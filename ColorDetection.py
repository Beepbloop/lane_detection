import numpy as np

# Mask is a zero matrix
# fillMask fills an empty numpy array with 255 for pixels that fits inside the defined triangle
def fill_color_mask(img):

    mask = np.zeros_like(img)
    h = mask.shape[0]
    w = mask.shape[1]

    bottom_left = (h, 0)
    middle = (int(h/2), int(w/2))
    bottom_right = (h, w)

    for x, row in enumerate(mask):
        for y, col in enumerate(row):
            # Applying equations to left_bound and right_bound
            left_bound = (h - x) * middle[1] / middle[0]
            right_bound = x * middle[1] / middle[0]
            if y > left_bound and y < right_bound and x <= 400:
                mask[x][y][0] = 255 #red
                mask[x][y][1] = 255 #green
                mask[x][y][2] = 255 #blue

    return mask

# For each non-zero pixel in mask, the corresponding pixel on image is kept (the rest of the pixels in mask is discarded)
def apply_mask_color(image, mask):
    return np.ma.array(image,mask=mask == 0).filled(fill_value=0)

def rgb_to_hsv(rgb):

    hsvImage = np.zeros_like(rgb, dtype='float64')

    redPrime = rgb[:,:,0] / 255
    greenPrime = rgb[:,:,1] / 255
    bluePrime = rgb[:,:,2] / 255

    cMax = np.maximum(np.maximum(redPrime, greenPrime), bluePrime)
    cMin = np.minimum(np.minimum(redPrime, greenPrime), bluePrime)
    delta = cMax - cMin

    #blues are the third third of the color wheel
    blue_max_idx = np.logical_and(cMax == bluePrime, delta != 0)
    hsvImage[blue_max_idx, 0] = (60 * ((redPrime[blue_max_idx] - greenPrime[blue_max_idx]) / delta[blue_max_idx]) + 240) % 360

    #greens are the second third of the color wheel
    green_max_idx = np.logical_and(cMax == greenPrime, delta != 0)
    hsvImage[green_max_idx, 0] = (60 * ((bluePrime[green_max_idx] - redPrime[green_max_idx]) / delta[green_max_idx]) + 120) % 360

    #reds are the first third of the color wheel
    red_max_idx = np.logical_and(cMax == redPrime, delta != 0)
    hsvImage[red_max_idx, 0] = (60 * ((greenPrime[red_max_idx] - bluePrime[red_max_idx]) / delta[red_max_idx]) + 360) % 360

    #if there is no net hue (no dominating color) of a pixel, set the hue to 0
    hsvImage[delta == 0, 0] = 0

    #saturation (lower value is more white, higher value is more colored (0 <= saturation <= 1000))
    hsvImage[cMax != 0, 1] = (delta[cMax != 0] / cMax[cMax != 0]) * 1000

    #value (lower value is more black, higher value is more colored (0 <= saturation <= 1000))
    hsvImage[:,:,2] = cMax * 1000 #in permille

    return hsvImage

def get_color(hue, saturation, value):
    if value < 500:
        return "darkColor"
    elif saturation <= 200:
        return "white"
    elif hue <= 300: #most pixels have a very low hue, so this must be fixed since it's counting the pavement
        return "red"
    elif hue <= 800:
        return "yellow"
