# Mask is a zero matrix
# fillMask fills an empty numpy array with 255 for pixels that fits inside the defined triangle
def fillMaskColor(mask, h, w):
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


    #
    # for x, row in enumerate(mask):
    #     for y, col in enumerate(row):
    #         if mask[x][y][0] != 255:
    #             image[x][y][0] = 0
    #             image[x][y][1] = 0
    #             image[x][y][2] = 0
    return np.ma.array(image,mask=mask == 0).filled(fill_value=0)
