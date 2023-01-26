import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image



def gamma_transform(im, c, g):
    '''
    This function performs gamma transformation on all values within an image
    Inputs:
        im: Image (np.array). In formula 3-5 this would be "r"
        c: Scalar constant
        g: Power constant
    
    Outputs:
        s: Transformed image
    '''
    ## Determine the input datatype
    if isinstance(im, np.ndarray):
        typ = type(im.ravel()[0])
    else: 
        typ = type(im)

    ## Perform the gamma transformation
    s = c*np.power(im, g)

    ## Force the output to have the original data type
    s = s.astype(typ)
    return s

def alt_transform(im, g):
    '''
    Alternative gamma transform that transforms value from the range [0,255]
    into the same value range.

    The basic Gamma transform can be changed to meet this requirement by only
    changing the value of 'c' in the equation: 
        T(s) = c*r^gamma

    You can find this by solving for 'c' in the equation: 
        255 = c*r^gamma

    Inputs:
        im: Image (np.array)
        g: Power constant

    Outputs:
        s: Transformed image
    '''

    ## Perform the transform    
    s = np.power(255, 1-g) * np.power(im, g)

    ## Force the datatype to uint8
    s = s.astype(np.uint8)

    return s



'''
Part (a)
'''
## Load the image
path = './Data/Fig0308(a)(fractured_spine).tif'
img = Image.open(path)
img = np.asarray(img)

## Perform the gamma transformations using c=1 and gamma=[0.6,0.4,0.3]
img_6a = gamma_transform(img, 1, 0.6)
img_4a = gamma_transform(img, 1, 0.4)
img_3a = gamma_transform(img, 1, 0.3)

## Print the value ranges of the original and transformed images.
print(f"r is in range [{img.min()}, {img.max()}]")
print(f"s is in range [{img_6a.min()}, {img_6a.max()}] when gamma = 0.6")
print(f"s is in range [{img_4a.min()}, {img_4a.max()}] when gamma = 0.4")
print(f"s is in range [{img_3a.min()}, {img_3a.max()}] when gamma = 0.3")

## Plot the original and transformed images
fig1, ax = plt.subplots(2,2, figsize=(10,10))
ax[0,0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0,0].set_title('Original Image')

ax[0,1].imshow(img_6a, cmap='gray', vmin=0, vmax=255)
ax[0,1].set_title('gamma = 0.6')

ax[1,0].imshow(img_4a, cmap='gray', vmin=0, vmax=255)
ax[1,0].set_title('gamma = 0.4')

ax[1,1].imshow(img_3a, cmap='gray', vmin=0, vmax=255)
ax[1,1].set_title('gamma = 0.3')

for a in ax.ravel():
    a.set_axis_off()

fig1.suptitle('Basic Gamma Transform')
plt.tight_layout()




'''
Part (b)
'''
## Perform the alternative transformation
img_6b = alt_transform(img, 0.6)
img_4b = alt_transform(img, 0.4)
img_3b = alt_transform(img, 0.3)

fig2, ay = plt.subplots(2,2, figsize=(10,10))
ay[0,0].imshow(img, cmap='gray', vmin=0, vmax=255)
ay[0,0].set_title('Original Image')

ay[0,1].imshow(img_6b, cmap='gray', vmin=0, vmax=255)
ay[0,1].set_title('gamma = 0.6')

ay[1,0].imshow(img_4b, cmap='gray', vmin=0, vmax=255)
ay[1,0].set_title('gamma = 0.4')

ay[1,1].imshow(img_3b, cmap='gray', vmin=0, vmax=255)
ay[1,1].set_title('gamma = 0.3')

for a in ay.ravel():
    a.set_axis_off()

fig2.suptitle('Alternate Gamma Transform')
plt.tight_layout()


plt.show()