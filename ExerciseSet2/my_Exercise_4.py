from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



path = './Data/Fig0310(b)(washed_out_pollen_image).tif'
im = np.array(Image.open(path))

##### Task (a)

def T(r,r0,e):
	return (r/r0)**e / (1 + (r/r0)**e)


x = np.linspace(0,1,100)
y1 = T(x,0.5,2)
y2 = T(x,0.5,4)
y3 = T(x,0.5,6)
y4 = T(x,0.5,8)


fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(6,8))

ax[0,0].plot(x,y1)
ax[0,0].plot(x,y2)
ax[0,0].plot(x,y3)
ax[0,0].plot(x,y4)
ax[0,0].legend(["E=2","E=4","E=6","E=8"])
ax[0,0].set_xlabel("r")
ax[0,0].set_ylabel("s")


##### Task (b)


def T_im(r,r0,e):
	## Convert the image to floats in range [0,1]
	_r = r.astype(float)/255

	## Compute the transformation
	# g = (_r/r0)**e / (1 + (_r/r0)**e)
	g = T(_r,r0,e)

	## Convert the transformed image back to the range [0,255] into the uint8 data type
	g = (g*255).astype(np.uint8)
	return g


plot01 = ax[0,1].imshow(im, cmap="gray")
plot10 = ax[1,0].imshow(T_im(im,0.5,2), cmap="gray")
plot11 = ax[1,1].imshow(T_im(im,0.5,4), cmap="gray")
plot20 = ax[2,0].imshow(T_im(im,0.5,6), cmap="gray")
plot21 = ax[2,1].imshow(T_im(im,0.5,8), cmap="gray")

plt.colorbar(plot01, ax=ax[0,1])
plt.colorbar(plot10, ax=ax[1,0])
plt.colorbar(plot11, ax=ax[1,1])
plt.colorbar(plot20, ax=ax[2,0])
plt.colorbar(plot21, ax=ax[2,1])

ax[0,0].set_title("Transform Plots")
ax[0,1].set_title("Original")
ax[1,0].set_title("E = 2")
ax[1,1].set_title("E = 4")
ax[2,0].set_title("E = 6")
ax[2,1].set_title("E = 8")


for a in ax.ravel():
	a.set_axis_off()
ax[0,0].set_axis_on()

plt.tight_layout()
plt.show()