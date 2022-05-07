import numpy as np
# ------------------------------------------------------ Cross correlation


def cross_correlation(image, filter, mode='valid'):
    if mode == 'valid':
        return valid_cross_correlation(image, filter)
    elif mode == 'full':
        return full_cross_correlation(image, filter)
    elif mode == 'same':
        return same_cross_correlation(image, filter)


def valid_cross_correlation(image, filter):
    return center_and_sum(image, filter)


def full_cross_correlation(image, filter):
    filter_row, filter_col = filter.shape
    h_pad = 2*int(filter_row/2)
    v_pad = 2*int(filter_col/2)

    image_padded = np.pad(
        image, ((h_pad, h_pad), (v_pad, v_pad)), 'constant', constant_values=(0, 0))
    return center_and_sum(image_padded, filter)


def same_cross_correlation(image, filter):
    filter_row, filter_col = filter.shape
    h_pad = int(filter_row/2)
    v_pad = int(filter_col/2)

    image_padded = np.pad(
        image, ((h_pad, h_pad), (v_pad, v_pad)), 'constant', constant_values=(0, 0))
    return center_and_sum(image_padded, filter)


# ----------------------------- helping method
def center_and_sum(image, filter):
    # new image
    filtered_image = []

    # The following vars are to determine the start of loop boundray + the slicing boundaries
    fil_row, fil_col = filter.shape
    row_start = int(fil_row/2)
    col_start = int(fil_col/2)

    # The following vars are to determine the end of loop boundry
    img_row, img_col = image.shape

    for i in range(row_start, img_row-row_start):
        # add new row with each row of the original image
        filtered_image.append([])
        for j in range(col_start, img_col-col_start):
            # [i-row_start] because we start at index [x,x] in the original image, which corresponds to [0,0] in the new image
            filtered_image[i-row_start].append(
                sum(sum(filter*image[i-row_start:i+row_start+1, j-col_start:j+col_start+1])))
    return np.array(filtered_image)


#------------------------------------------------------ Convolution

def convolution(image, filter, mode='valid'):
    filter = np.flip(filter)
    return cross_correlation(image, filter, mode=mode)
