import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

def transform(r, r0, E):
    numerator = np.power((r/r0), E)
    denomator = 1 + np.power((r/r0), E)
    t = numerator / denomator
    return t


def alt_transform(r, r0, E):
    '''
    The equation provided in Task 4 is bound to the range [0,1]. As r
    approaches infinity, T(r) will approach 1.  In order to force this
    transformation output to match the scale [0,255], a simple multiplication
    is applied.
    '''
    t = 255 * transform(r,r0,E)
    t = t.astype(np.uint8)
    return t




'''
Part (a)
'''
## Load the image
path = './Data/Fig0310(b)(washed_out_pollen_image).tif'
img = Image.open(path)

## Convert the image to a float and force it into the range [0,1]
img = np.asarray(img).astype(np.float64)
img = img/255 
print(f"r is in range [{np.min(img)}, {np.max(img)}]")

## Plot the original image
fig1, ax = plt.subplots(3,2, figsize=(10,10))
ax[0,0].imshow(img, cmap='gray', vmin=0, vmax=1)
ax[0,0].set_title('Original Image')

## Plot the transformation curves
x = np.linspace(0,1,1000)

ax[0,1].plot(x, transform(x, 0.5, 2), label="E=2")
ax[0,1].plot(x, transform(x, 0.5, 4), label="E=4")
ax[0,1].plot(x, transform(x, 0.5, 6), label="E=6")
ax[0,1].plot(x, transform(x, 0.5, 8), label="E=8")
ax[0,1].set_title("Transformation curves")
ax[0,1].legend()


## Plot the transformed images
ax[1,0].imshow(transform(img, 0.5, 2), cmap='gray', vmin=0, vmax=1)
ax[1,0].set_title('E = 2')

ax[1,1].imshow(transform(img, 0.5, 4), cmap='gray', vmin=0, vmax=1)
ax[1,1].set_title('E = 4')

ax[2,0].imshow(transform(img, 0.5, 6), cmap='gray', vmin=0, vmax=1)
ax[2,0].set_title('E = 6')

ax[2,1].imshow(transform(img, 0.5, 8), cmap='gray', vmin=0, vmax=1)
ax[2,1].set_title('E = 8')

for a in ax.ravel():
    a.set_axis_off()
ax[0,1].set_axis_on()

fig1.suptitle('Image range [0, 1]')
plt.tight_layout()





'''
Part (b)
'''
## Load the image
path = './Data/Fig0310(b)(washed_out_pollen_image).tif'
img = Image.open(path)

## Convert the image to a float but keep the [0,255] range
img = np.asarray(img).astype(np.float64)
print(f"r is in range [{np.min(img)}, {np.max(img)}]")

## Plot the original image
fig2, ay = plt.subplots(3,2, figsize=(10,10))
ay[0,0].imshow(img, cmap='gray', vmin=0, vmax=255)
ay[0,0].set_title('Original Image')

## Make lists of values of E and r0 to use
#  feel free to try different combinations here
Es = [4,4,8,8]
r0 = [128,200,128,200]

## Plot the transformation curves of the alternate
x = np.linspace(0,255,1000)

ay[0,1].plot(x, alt_transform(x, r0[0], Es[0]), label=f"E={Es[0]},r0={r0[0]}")
ay[0,1].plot(x, alt_transform(x, r0[1], Es[1]), label=f"E={Es[1]},r0={r0[1]}")
ay[0,1].plot(x, alt_transform(x, r0[2], Es[2]), label=f"E={Es[2]},r0={r0[2]}")
ay[0,1].plot(x, alt_transform(x, r0[3], Es[3]), label=f"E={Es[3]},r0={r0[3]}")
ay[0,1].set_title("Transformation curves")
ay[0,1].legend()

## Plot the images transformed using the alternate transform
ay[1,0].imshow(alt_transform(img, r0[0], Es[0]), cmap='gray', vmin=0, vmax=255)
ay[1,0].set_title(f"E={Es[0]}, r0={r0[0]}")

ay[1,1].imshow(alt_transform(img, r0[1], Es[1]), cmap='gray', vmin=0, vmax=255)
ay[1,1].set_title(f"E={Es[1]}, r0={r0[1]}")

ay[2,0].imshow(alt_transform(img, r0[2], Es[2]), cmap='gray', vmin=0, vmax=255)
ay[2,0].set_title(f"E={Es[2]}, r0={r0[2]}")

ay[2,1].imshow(alt_transform(img, r0[3], Es[3]), cmap='gray', vmin=0, vmax=255)
ay[2,1].set_title(f"E={Es[3]}, r0={r0[3]}")

for a in ay.ravel():
    a.set_axis_off()
ay[0,1].set_axis_on()

fig2.suptitle('Image range [0, 255]')
plt.tight_layout()
plt.show()