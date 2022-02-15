import numpy as np 
from scipy.signal import convolve2d
import matplotlib.pyplot as plt 
from PIL import Image

img = np.asarray(Image.open('Data/moon.tif'))
print(img.shape)


laplacian_kernel_4 = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

laplacian_kernel_8 = np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]])

img_4 = convolve2d(img, laplacian_kernel_4, mode='same')
img_8 = convolve2d(img, laplacian_kernel_8, mode='same')


fig, ax = plt.subplots(2,2,figsize=(20,10))
ax[0,0].imshow(img, cmap='gray')
ax[0,1].imshow(img_8, cmap='gray')
ax[1,0].imshow(img - img_4, cmap='gray', vmin=0, vmax=255)
ax[1,1].imshow(img - img_8, cmap='gray', vmin=0, vmax=255)

plt.show()