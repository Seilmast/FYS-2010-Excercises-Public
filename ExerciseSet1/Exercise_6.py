import scipy.io as io 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import NearestNDInterpolator 

img1 = io.loadmat('ExerciseSet1\Data\IM1.mat').get('IM1')
img2 = io.loadmat('ExerciseSet1\Data\IM2.mat').get('IM2')


def nearest_neighbour(im, zoom_factor):
    x_size, y_size = int(im.shape[0]*zoom_factor), int(im.shape[1]*zoom_factor)
    new_im = np.zeros((x_size, y_size))

    #Both x and y goes from some number n. We now want to increase this number to m. 
    #We must therefore map each discrete n to an m corresponding with it. We can do this
    #as such.
    x, y = np.linspace(0, (im.shape[0]-1), num=x_size), np.linspace(0, (im.shape[1]-1), num=y_size) 
    x, y = np.round(x, 0).astype(int), np.round(y, 0).astype(int)
    im = np.pad(im, pad_width=[(1,1), (1,1)])

    for old_x, new_x in zip(x, range(x_size)):
        for old_y, new_y in zip(y, range(y_size)):
            I = np.sum(im[old_x:(old_x+2), old_y:(old_y+2)])
            new_im[new_x, new_y] = I
    
    return new_im


    
def linear(im, zoom_factor):
    x_size, y_size = int(im.shape[0]*zoom_factor), int(im.shape[1]*zoom_factor)
    new_im = np.zeros((x_size, y_size))

    #Both x and y goes from some number n. We now want to increase this number to m. 
    #We must therefore map each discrete n to an m corresponding with it. We can do this
    #as such.
    x, y = np.linspace(0, (im.shape[0]-1), num=x_size), np.linspace(0, (im.shape[1]-1), num=y_size) 
    x, y = np.round(x, 0).astype(int), np.round(y, 0).astype(int)

    for old_x, new_x in zip(x, range(x_size)):
        for old_y, new_y in zip(y, range(y_size)):
            I = im[old_x, old_y]
            new_im[new_x, new_y] = I

    return new_im 

def bicubic(im, zoom_factor):
    x_size, y_size = int(im.shape[0]*zoom_factor), int(im.shape[1]*zoom_factor)
    new_im = np.zeros((x_size, y_size))

    #Both x and y goes from some number n. We now want to increase this number to m. 
    #We must therefore map each discrete n to an m corresponding with it. We can do this
    #as such.
    x, y = np.linspace(0, (im.shape[0]-1), num=x_size), np.linspace(0, (im.shape[1]-1), num=y_size) 
    x, y = np.round(x, 0).astype(int), np.round(y, 0).astype(int)
    im = np.pad(im, pad_width=[(2,2), (2,2)])

    for old_x, new_x in zip(x, range(x_size)):
        for old_y, new_y in zip(y, range(y_size)):
            I = np.sum(im[old_x:(old_x+3), old_y:(old_y+3)])
            new_im[new_x, new_y] = I
    
    return new_im


fig, ax = plt.subplots(2,2,figsize=(10,10))

ax[0,0].imshow(img2, cmap='gray')
ax[0,0].set_title('Original Image')

ax[0,1].imshow(linear(img2, 10), cmap='gray')
ax[0,1].set_title('Linear')

ax[1,0].imshow(nearest_neighbour(img2, 10), cmap='gray')
ax[1,0].set_title('Nearest Neigbour')

ax[1,1].imshow(bicubic(img2, 10), cmap='gray')
ax[1,1].set_title('Bicubic')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
plt.close()