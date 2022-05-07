
import numpy as np
import scipy.signal as ssig
from skimage import io, filters
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import numpy as np
    import scipy.signal as ssig
    from image_processing import *
    
    # sobel filter to detect edges -> acts like dervative
    filter = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    image = io.imread('shrek.png', as_gray=True) # read image as grayscale

    edges = cross_correlation(image, filter, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[1].imshow(edges, cmap=plt.cm.gray)
    plt.show()
