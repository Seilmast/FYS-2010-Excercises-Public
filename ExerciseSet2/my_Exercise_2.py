from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



def im2double(im):

	## Convert the data type to float
	im = im.astype(float)

	## Scale the intensities to [0,1]
	im /= 255

	return im


def im2uint8(im):

	## Convert the data type to float
	im = im.astype(float)

	## Scale the intensities to [0,255]
	im *= 255

	return im


impath = "./Data/Fig0310(b)(washed_out_pollen_image).tif"
I = Image.open(impath)
arr_int = np.array(I)
# arr_dbl = arr_int.astype(float)/255

# print(arr_int.min(), arr_int.max())
# print(arr_dbl.min(), arr_dbl.max())

arr_dbl = im2double(arr_int)

print(arr_dbl.min(), arr_dbl.max())

plt.imshow(arr_dbl, cmap="gray")
plt.colorbar()
# plt.show()

