import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


## Read the image
img = cv2.imread("Fig1016(a)(building_original).tif")

## Initialize the plots
fig,ax = plt.subplots(1,3, figsize=(14,4))
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original")

'''
PART A
'''
## Define LoG parameters
log_ksize = 11
sig = 1

## Define LoG kernel
x = np.arange(log_ksize)
x = x - x.mean()
y = np.copy(x)
x,y = np.meshgrid(x,y)

coef = (x**2 + y**2 - 2*sig**2)/sig**4
exponent = np.exp(-1 * (x**2 + y**2) / (2*sig**2))
log_kernel = coef * exponent

## Apply LoG kernel
log_edges = cv2.filter2D(img, ddepth=-1, kernel=log_kernel)

## Plot the LoG edges
ax[1].imshow(log_edges, cmap="gray")
ax[1].set_title("LoG Edges")


'''
PART B
'''
canny_ksize = 9
t_lo = 100
t_hi = 200

canny_edges = cv2.Canny(img, t_lo, t_hi, canny_ksize)
ax[2].imshow(canny_edges, cmap="gray")
ax[2].set_title("Canny Edges")



'''
PART C
'''
## Load the image
blaklokke = cv2.imread("blaklokke.jpg", cv2.IMREAD_GRAYSCALE)

## Threshold the image with OpenCV's Otsu method
otsu_thresh_val,otsu_binary = cv2.threshold(blaklokke, 0, 255, cv2.THRESH_OTSU)

## Plot the image
figb,axb = plt.subplots(1,2, figsize=(7,4))
axb[0].imshow(blaklokke, cmap="gray")
axb[0].set_title("Original")
axb[1].imshow(otsu_binary, cmap="gray")
axb[1].set_title("Otsu Thresholded Image")


'''
Format the plots
'''
plt.figure(figb.number)
for a in axb.ravel():
    a.set_axis_off()
plt.tight_layout()

plt.figure(fig.number)
for a in ax.ravel():
    a.set_axis_off()
plt.tight_layout()

plt.show()
