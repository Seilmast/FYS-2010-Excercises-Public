import numpy as np 
from scipy.signal import convolve2d
import matplotlib.pyplot as plt 
from PIL import Image

img = np.asarray(Image.open('./Data/test_pattern.tif'))

kernel_size = 15
def avg_kernel(kernel_size):
    return np.ones((kernel_size, kernel_size))/(kernel_size**2)

fig, ax = plt.subplots(2,2,figsize=(10,10))
ax[0,0].imshow(img, cmap='gray')
ax[0,0].set_title('Input')
ax[0,1].imshow(convolve2d(img, avg_kernel(3), mode='same'), cmap='gray')
ax[0,1].set_title('Avg kernel of size 3')
ax[1,0].imshow(convolve2d(img, avg_kernel(5), mode='same'), cmap='gray')
ax[1,0].set_title('Avg kernel of size 5')
ax[1,1].imshow(convolve2d(img, avg_kernel(9), mode='same'), cmap='gray')
ax[1,1].set_title('Avg kernel of size 9')
plt.show()