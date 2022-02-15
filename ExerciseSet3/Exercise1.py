import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter

def checkerboard(shape):
    return np.indices(shape).sum(axis=(0)) % 2


img_binary = np.zeros((1600, 1600))
img_binary[:,:799] = 255
img_checkerboard = np.zeros((1600, 1600))

for row in range(img_checkerboard.shape[0]):
    for col in range(img_checkerboard.shape[1]):

        checker_size = 200   #1600 / checker_size should be int
        r, c = (row // checker_size) % 2 == 0, (col // checker_size) % 2 == 0
        
        #if r & c is even, then black, else white
        if r != c:
            img_checkerboard[row, col] = 255

sigma = 9

blurred_binary = gaussian_filter(img_binary, sigma=sigma)
blurred_checkerboard = gaussian_filter(img_checkerboard, sigma=sigma)



fig, ax = plt.subplots(3,2, figsize=(10,10))
ax[0,0].imshow(img_binary, cmap='gray')
ax[0,0].set_title('Binary Image')
ax[0,1].imshow(img_checkerboard, cmap='gray')
ax[0,1].set_title('Checkerboard Image')

ax[1,0].hist(img_binary.flatten(), 255)
ax[1,0].set_title('Binary Image Histogram')
ax[1,1].hist(img_checkerboard.flatten(), 255)
ax[1,1].set_title('Checkerboard Image Histogram')

ax[2,0].hist(blurred_binary.flatten(), 255)
ax[2,0].set_title('Blurred Binary Image Histogram')
ax[2,1].hist(blurred_checkerboard.flatten(), 255)
ax[2,1].set_title('Blurred Checkerboard Image Histogram')

plt.tight_layout()
plt.show()