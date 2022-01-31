import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

def transform(r, r0, E):
    t = np.power((r/r0), E)/(1 + np.power((r/r0), E))
    return t

def new_range(im, min_val = 0, max_val = 1.0):
    im_min, im_max = np.min(im), np.max(im)
    new_min, new_max = min_val, max_val

    im_range = im_max - im_min 
    new_range = new_max - new_min


    t = ((im - im_min)*new_range / im_range) + new_min
    return t

path = 'Data\Fig0310(b)(washed_out_pollen_image).tif'
img = Image.open(path)
img = np.asarray(img).astype(np.float64)
img = np.copy(img/255) 

im_min, im_max = np.min(img), np.max(img)
print(f"r is in range [{im_min}, {im_max}]")

fig1, ax = plt.subplots(3,2, figsize=(10,10))
ax[0,0].imshow(img, cmap='gray')
ax[0,0].set_title('Original Image')

ax[0,1].imshow(new_range(img), cmap='gray')
ax[0,1].set_title('Transformed to range [0,1]')

#Convert image to [0,1] range
img = np.copy(new_range(img))

ax[1,0].imshow(transform(img, 0.5, 2), cmap='gray')
ax[1,0].set_title('E = 2')

ax[1,1].imshow(transform(img, 0.5, 4), cmap='gray')
ax[1,1].set_title('E = 4')

ax[2,0].imshow(transform(img, 0.5, 6), cmap='gray')
ax[2,0].set_title('E = 6')

ax[2,1].imshow(transform(img, 0.5, 8), cmap='gray')
ax[2,1].set_title('E = 8')

for a in ax.ravel():
    a.set_axis_off()

fig1.suptitle('Image range [0, 255]')
plt.tight_layout()

##### Using range [0, 255] below #####

path = 'Data\Fig0310(b)(washed_out_pollen_image).tif'
img = Image.open(path)
img = np.asarray(img).astype(np.float64)


im_min, im_max = np.min(img), np.max(img)
print(f"r is in range [{im_min}, {im_max}]")

fig2, ay = plt.subplots(3,2, figsize=(10,10))
ay[0,0].imshow(img, cmap='gray')
ay[0,0].set_title('Original Image')

ay[0,1].imshow(new_range(img), cmap='gray')
ay[0,1].set_title('Transformed to range [0,255]')

#Convert image to [0,255] range
img = np.copy(new_range(img, max_val=255))

ay[1,0].imshow(transform(img, 128, 2), cmap='gray')
ay[1,0].set_title('E = 2')

ay[1,1].imshow(transform(img, 128, 4), cmap='gray')
ay[1,1].set_title('E = 4')

ay[2,0].imshow(transform(img, 128, 6), cmap='gray')
ay[2,0].set_title('E = 6')

ay[2,1].imshow(transform(img, 128, 8), cmap='gray')
ay[2,1].set_title('E = 8')

for a in ay.ravel():
    a.set_axis_off()

fig2.suptitle('Image range [0, 255]')
plt.tight_layout()
plt.show()