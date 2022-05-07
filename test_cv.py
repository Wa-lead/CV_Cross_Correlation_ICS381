if __name__ == "__main__":
    import numpy as np
    import scipy.signal as ssig
    from image_processing import *
    
    # odd sized filter and image
    filter = (1/10)*np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])
    image = np.array([[5,1,9,3,3],
                      [7,9,2,5,7],
                      [1,7,2,8,6],
                      [3,6,4,8,1],
                      [4,9,3,3,6]])
                    
    # check cross-correlation
    full_res = cross_correlation(image, filter, mode='full')
    valid_res = cross_correlation(image, filter, mode='valid')
    same_res = cross_correlation(image, filter, mode='same')
    print('Checking cross-correlation\n')
    print(full_res, full_res.shape)
    print('_______________________________________________________________________\n')
    print(valid_res, valid_res.shape)
    print('_______________________________________________________________________\n')
    print(same_res, same_res.shape)
    print('_______________________________________________________________________\n')
    
    # check assert that you get same results as scipy's implementation
    assert np.array_equal(full_res.round(4), ssig.correlate2d(image, filter, mode='full').round(4))
    assert np.array_equal(valid_res.round(4), ssig.correlate2d(image, filter, mode='valid').round(4))
    assert np.array_equal(same_res.round(4), ssig.correlate2d(image, filter, mode='same').round(4))
    
    # check convolution
    full_res = convolution(image, filter, mode='full')
    valid_res = convolution(image, filter, mode='valid')
    same_res = convolution(image, filter, mode='same')
    print('Checking convolution\n')
    print(full_res, full_res.shape)
    print('_______________________________________________________________________\n')
    print(valid_res, valid_res.shape)
    print('_______________________________________________________________________\n')
    print(same_res, same_res.shape)
    print('_______________________________________________________________________\n')
    
    # check assert that you get same results as scipy's implementation
    assert np.array_equal(full_res.round(4), ssig.convolve(image, filter, mode='full').round(4))
    assert np.array_equal(valid_res.round(4), ssig.convolve(image, filter, mode='valid').round(4))
    assert np.array_equal(same_res.round(4), ssig.convolve(image, filter, mode='same').round(4))