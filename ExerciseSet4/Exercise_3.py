import numpy as np 
from scipy.signal import convolve2d
from PIL import Image 
import matplotlib.pyplot as plt 
img = Image.open('FYS-2010-Excercises-Public/ExerciseSet3/Data/DIP3E_CH03_Original_Images/DIP3E_Original_Images_CH03/Fig0333(a)(test_pattern_blurring_orig).tif')
img = np.asarray(img)

k = [3,5,11]


#Averaging filter

fig, ax = plt.subplots(1,4)
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Unfiltered')
for i, kernel in enumerate(k):
    avg_filter = np.ones((kernel,kernel)) / kernel**2 
    img_f = convolve2d(img, avg_filter, mode='same')
    ax[i+1].imshow(img_f, cmap='gray')
    ax[i+1].set_title(f'Kernel size {kernel}')

plt.show()

img = Image.open('FYS-2010-Excercises-Public/ExerciseSet3/Data/DIP3E_CH03_Original_Images/DIP3E_Original_Images_CH03/Fig0338(a)(blurry_moon).tif')
img = np.asarray(img)

laplacian1 = np.array([[0, -1, 0], 
                       [-1, 4, -1], 
                       [0, -1, 0]])
laplacian2 = np.array([[-1, -1, -1], 
                       [-1, 8, -1], 
                       [-1, -1, -1]])

fig, ax = plt.subplots(2,3)
ax[0,0].imshow(img, cmap='gray')
ax[0,1].imshow(convolve2d(img, laplacian1, mode='same'), cmap='gray')
ax[0,2].imshow(convolve2d(img, laplacian2, mode='same'), cmap='gray')

ax[1,0].imshow(img, cmap='gray')
ax[1,1].imshow(img + convolve2d(img, laplacian1, mode='same'), cmap='gray', vmin=0, vmax=255)
ax[1,2].imshow(img + convolve2d(img, laplacian2, mode='same'), cmap='gray', vmin=0, vmax=255)
plt.show()