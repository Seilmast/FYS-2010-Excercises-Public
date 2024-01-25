import numpy as np
from matplotlib import pyplot as plt



def intensity_function(n_bits):
    '''
    This function computes the intensity equation from Task 4, then reduces the
    digital resolution to n_bits.
    '''

    ## Define the x,y grid
    grid_size = 1000
    grid_range = np.linspace(0,4,grid_size)
    x,y = np.meshgrid(grid_range, grid_range)
    x0 = 2
    y0 = 2

    ## Compute the intensity
    i = 255 * np.exp(-( (x-x0)**2 + (y-y0)**2 ))

    ## Force the intenisty to be an 8-bit integer
    i = np.array(i).astype(np.uint8)

    ## Mask the lowest n_bits to reduce the digital resolution
    bitmask = int("1"*(n_bits) + "0"*(8-n_bits), base=2)
    # print("1"*(n_bits) + "0"*(8-n_bits))
    i = np.bitwise_and(i, bitmask)

    return i


## Initialize a plot to display the options
fig,ax = plt.subplots(2,4, figsize=(14,7))
ax = ax.ravel()

for bit in np.linspace(8,1,8, dtype=int):
    img = intensity_function(bit)
    ax[bit-1].imshow(img, cmap="gray")
    ax[bit-1].set_axis_off()
    ax[bit-1].set_title(f"{bit} bits")

plt.tight_layout()
plt.show()

