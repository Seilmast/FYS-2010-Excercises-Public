import enum
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## Load the image as an array
img = np.array(Image.open("Fig0630(01)(strawberries_fullcolor).tif")) / 255
# img = img[:,:-1]


#########################################################
###################### Subtask (a) ######################
#########################################################

## Define the strawberry subregion with a bounding box found from looking 
## at the plot of img.
bb = ((70,0),(360,360))
strawberry_region = img[bb[0][0]:bb[1][0], bb[0][1]:bb[1][1]]

# Plot the image and subregion
fig1,ax1 = plt.subplots(nrows=1, ncols=2)
ax1[0].imshow(img)
ax1[0].set_title("Full Image")
ax1[1].imshow(strawberry_region)
ax1[1].set_title("Strawberry Subregion")
for a in ax1:
    a.set_axis_off()

## Compute the mean and covariance of the color vectors
# First reshape the region to a list of color vectors
region_vectors = strawberry_region.reshape(-1,3).T      
a = np.mean(region_vectors, axis=1)
C = np.cov(region_vectors)  


#########################################################
###################### Subtask (b) ######################
#########################################################

## Compute the inverse of the covariance matrix
Ci = np.linalg.inv(C)

## Use a numpy broadcasting trick to quickly compute D for 
## every pixel in the image.
D = np.einsum("ijc,cji->ij", (img-a) @ Ci, (img-a).T)


## Plot the D value for each pixel
fig2,ax2 = plt.subplots(1,1)
dmap_plot = ax2.imshow(D)
fig2.colorbar(dmap_plot, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_title("D Map")


## Define a few thresholds for D0 and test them
def testD0(d_thresh):
    # Function that applies a D threshold to an image.
    seg_map = D<=d_thresh
    seg_im = img * seg_map[...,None]
    return seg_map, seg_im

D0 = [1,5,20,50]
segmentations = [testD0(d0_val) for d0_val in D0]

## Plot the segmentations
fig3,ax3 = plt.subplots(nrows=2, ncols=len(D0), figsize=(12,6))

for ix,seg in enumerate(segmentations):
    ax3[0,ix].imshow(seg[0])
    ax3[1,ix].imshow(seg[1])
    ax3[0,ix].set_title(f"D0 = {D0[ix]}")


# plt.tight_layout()
ax2.set_axis_off()
for a in ax3.ravel():
    a.set_axis_off()
plt.show()
