from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



def im2double(im):

	## Convert the data type to float
	_im = im.astype(np.float64)

	## Scale the intensities to [0,1]
	_im = _im/255

	return _im


def im2uint8(im):

	## Scale the intensities to [0,255]
	_im = im*255

	## Convert the data type to float
	_im = _im.astype(np.uint8)

	return _im


## Load the image
impath = "./Data/Fig0310(b)(washed_out_pollen_image).tif"
I = Image.open(impath)
arr = np.array(I)

## Create a double version of the image
arr_dbl = im2double(arr)
print("float64 image value range:", arr_dbl.min(), arr_dbl.max())

## Create a uint8 version of the image
arr_int = im2uint8(arr_dbl)
print("uint8 image value range:", arr_int.min(), arr_int.max())

## Display each image type with colorbars
fig,ax = plt.subplots(1,2, figsize=(10,5))
dblplt = ax[0].imshow(arr_dbl, cmap="gray")
plt.colorbar(dblplt, ax=ax[0])
ax[0].set_title("double image")
intplt = ax[1].imshow(arr_int, cmap="gray")
plt.colorbar(intplt, ax=ax[1])
ax[1].set_title("uint8 image")

for a in ax.ravel(): a.set_axis_off()

plt.show()

