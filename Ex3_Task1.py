import cv2 
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 
img = Image.open('FYS-2010-Excercises-Public/ExerciseSet3/Data/DIP3E_CH03_Original_Images/DIP3E_Original_Images_CH03/Fig0335(a)(ckt_board_saltpep_prob_pt05).tif')
img = np.asarray(img)

fig, ax = plt.subplots(2,3)
ax[0,0].imshow(img, cmap='gray')
ax[1,0].hist(img.flatten(), bins=256, density = True)
ax[0,0].set_title('Unfiltered')
img_f = cv2.medianBlur(img, 5)
ax[0,1].imshow(img_f, cmap='gray')
ax[1,1].hist(img_f.flatten(), bins=256, density = True)
ax[0,1].set_title('Kernel size 5')

img_f = cv2.medianBlur(img, 11)
ax[0,2].imshow(img_f, cmap='gray')
ax[1,2].hist(img_f.flatten(), bins=256, density = True)
ax[0,2].set_title('Kernel size 11')

plt.show()