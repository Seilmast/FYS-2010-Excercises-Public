from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def gammaTransform(im, c, gamma, epsilon=0):
	t_im = c*(im + epsilon)**gamma
	return t_im.astype(np.uint8)


def myTransform(im, c, gamma, epsilon=0):
	t_im = c*(im + epsilon)**gamma
	new_max = c*(255 + epsilon)**gamma
	t_im *= 255/new_max
	return t_im.astype(np.uint8)


'''
Reporoducing figures 3.8
'''
## Task (a)
path = './Data/Fig0308(a)(fractured_spine).tif'
im1 = np.array(Image.open(path)).astype(float)

fig1,ax1 = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
plot00 = ax1[0,0].imshow(im1, cmap="gray")
plot01 = ax1[0,1].imshow(gammaTransform(im1,1,0.6), cmap="gray")
plot10 = ax1[1,0].imshow(gammaTransform(im1,1,0.4), cmap="gray")
plot11 = ax1[1,1].imshow(gammaTransform(im1,1,0.3), cmap="gray")

ax1[0,0].set_title("Original")
ax1[0,1].set_title("c=1, gamma=0.6")
ax1[1,0].set_title("c=1, gamma=0.4")
ax1[1,1].set_title("c=1, gamma=0.3")

plt.colorbar(plot00, ax=ax1[0,0])
plt.colorbar(plot01, ax=ax1[0,1])
plt.colorbar(plot10, ax=ax1[1,0])
plt.colorbar(plot11, ax=ax1[1,1])

fig1.suptitle("gammaTransform")
for a in ax1.ravel():
	a.set_axis_off()

plt.tight_layout()


## Task (b)
fig2,ax2 = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
plot00 = ax2[0,0].imshow(im1, cmap="gray")
plot01 = ax2[0,1].imshow(myTransform(im1,1,0.6), cmap="gray")
plot10 = ax2[1,0].imshow(myTransform(im1,1,0.4), cmap="gray")
plot11 = ax2[1,1].imshow(myTransform(im1,1,0.3), cmap="gray")

ax2[0,0].set_title("Original")
ax2[0,1].set_title("c=1, gamma=0.6")
ax2[1,0].set_title("c=1, gamma=0.4")
ax2[1,1].set_title("c=1, gamma=0.3")

plt.colorbar(plot00, ax=ax2[0,0])
plt.colorbar(plot01, ax=ax2[0,1])
plt.colorbar(plot10, ax=ax2[1,0])
plt.colorbar(plot11, ax=ax2[1,1])

fig2.suptitle("myTransform")
for a in ax2.ravel():
	a.set_axis_off()

plt.tight_layout()

'''
Reporoducing figures 3.9
'''
## Task (a)
path = './Data/Fig0309(a)(washed_out_aerial_image).tif'
im2 = np.array(Image.open(path)).astype(float)

fig3,ax3 = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
plot00 = ax3[0,0].imshow(im2, cmap="gray")
plot01 = ax3[0,1].imshow(gammaTransform(im2,1,3.0), cmap="gray")
plot10 = ax3[1,0].imshow(gammaTransform(im2,1,4.0), cmap="gray")
plot11 = ax3[1,1].imshow(gammaTransform(im2,1,5.0), cmap="gray")

ax3[0,0].set_title("Original")
ax3[0,1].set_title("c=1, gamma=3.0")
ax3[1,0].set_title("c=1, gamma=4.0")
ax3[1,1].set_title("c=1, gamma=5.0")

plt.colorbar(plot00, ax=ax3[0,0])
plt.colorbar(plot01, ax=ax3[0,1])
plt.colorbar(plot10, ax=ax3[1,0])
plt.colorbar(plot11, ax=ax3[1,1])

fig3.suptitle("gammaTransform")
for a in ax3.ravel():
	a.set_axis_off()

plt.tight_layout()

## Task (b)
fig4,ax4 = plt.subplots(nrows=2, ncols=2, figsize=(6,6))
plot00 = ax4[0,0].imshow(im2, cmap="gray")
plot01 = ax4[0,1].imshow(myTransform(im2,1,3.0), cmap="gray")
plot10 = ax4[1,0].imshow(myTransform(im2,1,4.0), cmap="gray")
plot11 = ax4[1,1].imshow(myTransform(im2,1,5.0), cmap="gray")

ax4[0,0].set_title("Original")
ax4[0,1].set_title("c=1, gamma=3.0")
ax4[1,0].set_title("c=1, gamma=4.0")
ax4[1,1].set_title("c=1, gamma=5.0")

plt.colorbar(plot00, ax=ax4[0,0])
plt.colorbar(plot01, ax=ax4[0,1])
plt.colorbar(plot10, ax=ax4[1,0])
plt.colorbar(plot11, ax=ax4[1,1])

fig4.suptitle("myTransform")
for a in ax4.ravel():
	a.set_axis_off()

plt.tight_layout()

plt.show()