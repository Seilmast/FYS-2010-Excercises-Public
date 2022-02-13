import numpy as np

def new_range(im, min_val = 0, max_val = 1.0):
    '''
        im: Any matrix of intensities.
        min_val, max_val: define the new range going from [min_val, max_val].
    '''
    im_min, im_max = np.min(im), np.max(im)
    new_min, new_max = min_val, max_val

    im_range = im_max - im_min 
    new_range = new_max - new_min


    t = ((im - im_min)*new_range / im_range) + new_min
    return t

some_range = np.linspace(0, 100, num=100)
print(f'Initial range: {np.min(some_range)} to {np.max(some_range)}')
some_new_range = new_range(some_range, min_val=-10, max_val = 10)
print(f'Initial range: {np.min(some_new_range)} to {np.max(some_new_range)}')