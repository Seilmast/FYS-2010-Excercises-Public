import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#########################################################
###################### Subtask (a) ######################
#########################################################

## Load the image as an array
img = np.array(Image.open("blaklokke.jpg")) / 255

## Create subplots and plot the color image
fig_a,ax_a = plt.subplots(nrows=2, ncols=2, figsize=(7,7))
ax_a[0,0].imshow(img)
ax_a[0,0].set_title("Color Image")

## Plot the RGB channels of the image on the other subplots
ax_a[0,1].imshow(img[:,:,0], cmap="gray", vmin=0, vmax=1)
ax_a[0,1].set_title("Red Channel")

ax_a[1,0].imshow(img[:,:,1], cmap="gray", vmin=0, vmax=1)
ax_a[1,0].set_title("Green Channel")

ax_a[1,1].imshow(img[:,:,2], cmap="gray", vmin=0, vmax=1)
ax_a[1,1].set_title("Blue Channel")

## Turn off the plot axes
for a in ax_a.ravel():
    a.set_axis_off()


#########################################################
###################### Subtask (b) ######################
#########################################################

## Compute the histogram of each channel
fig_b,ax_b = plt.subplots(nrows=1, ncols=3, figsize=(9,3))

ax_b[0].hist(img[:,:,0].ravel(), bins=255)
ax_b[0].set_title("Red Channel")
ax_b[0].set_xlabel("Pixel Value")

ax_b[1].hist(img[:,:,1].ravel(), bins=255)
ax_b[1].set_title("Green Channel")
ax_b[1].set_xlabel("Pixel Value")

ax_b[2].hist(img[:,:,2].ravel(), bins=255)
ax_b[2].set_title("Blue Channel")
ax_b[2].set_xlabel("Pixel Value")

plt.tight_layout()


#########################################################
###################### Subtask (c) ######################
#########################################################

def rgb_to_hsi(im):
    '''Function to convert an RGB image to an HSI image.'''
    ## Scale the image down to the range [0,1]
    R,G,B = im[:,:,0],im[:,:,1],im[:,:,2]

    ## From equation 6-17
    theta = np.arccos( 0.5*((R-G)+(R-B)) / (np.sqrt((R-G)**2 + (R-B)*(G-B)) + 1e-8) )
    theta *= 180/np.pi          # Convert the angle into degrees

    ## From equation 6-16
    H = np.copy(theta)
    H[B>G] = 360-theta[B>G]

    ## From equation 6-18
    S = 1-3*np.min(im, axis=2)/np.sum(im, axis=2)

    ## From equation 6-19
    I = np.mean(im, axis=2)

    ## return HSI image
    return np.stack([H,S,I], axis=2)

## Plot HSI image
hsi_img = rgb_to_hsi(img)
fig_c, ax_c = plt.subplots(nrows=2, ncols=2, figsize=(7,7))
ax_c[0,0].imshow(hsi_img)
ax_c[0,0].set_title("HSI")

## Plot H,S,I channels
h = ax_c[0,1].imshow(hsi_img[:,:,0], cmap="gray")
ax_c[0,1].set_title("Hue")
fig_c.colorbar(h, ax=ax_c[0,1])

s = ax_c[1,0].imshow(hsi_img[:,:,1], cmap="gray")
ax_c[1,0].set_title("Saturation")
fig_c.colorbar(s, ax=ax_c[1,0])

i = ax_c[1,1].imshow(hsi_img[:,:,2], cmap="gray")
ax_c[1,1].set_title("Intensity")
fig_c.colorbar(i, ax=ax_c[1,1])

## Turn off the plot axes
for a in ax_c.ravel():
    a.set_axis_off()


#########################################################
###################### Subtask (d) ######################
#########################################################

## Compute the histogram of each channel
## Compute the histogram of each channel
fig_d,ax_d = plt.subplots(nrows=1, ncols=3, figsize=(9,3))

ax_d[0].hist(hsi_img[:,:,0].ravel(), bins=360)
ax_d[0].set_title("Hue Channel")
ax_d[0].set_xlabel("Pixel Value")

ax_d[1].hist(hsi_img[:,:,1].ravel(), bins=255)
ax_d[1].set_title("Saturation Channel")
ax_d[1].set_xlabel("Pixel Value")

ax_d[2].hist(hsi_img[:,:,2].ravel(), bins=255)
ax_d[2].set_title("Intensity Channel")
ax_d[2].set_xlabel("Pixel Value")

plt.tight_layout()


plt.show()