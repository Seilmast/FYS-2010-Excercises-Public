import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter


'''
Produce the two images
'''
def checkerboard(shape):
    return np.indices(shape).sum(axis=(0)) % 2

img_binary = np.zeros((1600, 1600))
img_binary[:,:799] = 255
img_checkerboard = np.zeros((1600, 1600))

for row in range(img_checkerboard.shape[0]):
    for col in range(img_checkerboard.shape[1]):

        checker_size = 200   #1600 / checker_size should be int
        r, c = (row // checker_size) % 2 == 0, (col // checker_size) % 2 == 0
        
        if r != c:
            img_checkerboard[row, col] = 255

'''
Blur the images
'''
sigma = 21
blurred_binary = gaussian_filter(img_binary, sigma=sigma)
blurred_checkerboard = gaussian_filter(img_checkerboard, sigma=sigma)


'''
Plot the original and blurred images with their histograms
'''
fig, ax = plt.subplots(2,4, figsize=(10,5))
ax[0,0].imshow(img_binary, cmap='gray')
ax[0,0].set_title('Binary Image')
ax[0,1].imshow(img_checkerboard, cmap='gray')
ax[0,1].set_title('Checkerboard Image')

ax[0,2].imshow(blurred_binary, cmap='gray')
ax[0,2].set_title('Blurred Binary Image')
ax[0,3].imshow(blurred_checkerboard, cmap='gray')
ax[0,3].set_title('Blurred Checkerboard Image')

ax[1,0].hist(img_binary.flatten(), 255)
ax[1,0].set_title('Binary Image Histogram')
ax[1,1].hist(img_checkerboard.flatten(), 255)
ax[1,1].set_title('Checkerboard Image Histogram')

ax[1,2].hist(blurred_binary.flatten(), 255)
ax[1,2].set_title('Blurred Binary Image \nHistogram')
ax[1,3].hist(blurred_checkerboard.flatten(), 255)
ax[1,3].set_title('Blurred Checkerboard \nImage Histogram')

for a in ax[0,:]: 
    a.set_xticks([])
    a.set_yticks([])

plt.tight_layout()
plt.show()