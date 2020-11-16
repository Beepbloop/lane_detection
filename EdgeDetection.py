import numpy as np

def smoothing(input_im, kernel_shape=(5,5)):
    # Tharm's image smoothing
    #Implementing image smoothing
    rows = input_im.shape[0]
    cols = input_im.shape[1]

    #input any kernel size
    kernel_shape_row = kernel_shape[0]
    kernel_shape_col = kernel_shape[1]

    kernel = np.ones((kernel_shape_row, kernel_shape_col))/(kernel_shape_row * kernel_shape_col)
    mag_smooth = np.zeros((rows,cols))

    #pad with zeros on the border for full blurring of image
    padded_gray = np.zeros((rows + kernel_shape_row - 1, cols + kernel_shape_col - 1))
    index_1_row = (int) ((kernel_shape_row - 1)/2)
    index_last_row = (int) (rows + ((kernel_shape_row - 1)/2))
    index_1_col = (int) ((kernel_shape_col - 1)/2)
    index_last_col = (int) (cols + ((kernel_shape_col - 1)/2))
    padded_gray[index_1_row:index_last_row, index_1_col:index_last_col] = input_im

    for x in range(rows):
        for y in range(cols):
            mag_smooth[x][y] = (kernel * padded_gray[x:x+kernel_shape_row, y:y+kernel_shape_col]).sum()

    return mag_smooth

def sobel_edge_detector(input_im):
    # Tharm's Sobel Edge Detector
    ## Implementing sobel edge detector

    rows = input_im.shape[0] - 1
    cols = input_im.shape[1] - 1

    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    mag_G = np.zeros((rows,cols))

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            sumx = (Gx * input_im[i-1:i+2, j-1:j+2]).sum()
            sumy = (Gy * input_im[i-1:i+2, j-1:j+2]).sum()
            mag_G[i][j] = sumx**2 + sumy**2

    return np.sqrt(mag_G)


# Rohit's Otsu's Implementation
def otsu_threshold(input_im):
    # Set total number of bins in the histogram
    bins_num = int(np.amax(input_im)) + 1
    print(f'Number of bins = {bins_num}')

    # Get the image histogram
    hist, bin_edges = np.histogram(input_im, bins=bins_num, range=(0, bins_num))

    # normalize histogram
    hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]

    return threshold

def apply_threshold(input_im, thresh):
    input_im[input_im > thresh] = 255
    input_im[input_im <= thresh] = 0
    return input_im
