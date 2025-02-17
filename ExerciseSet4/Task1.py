import numpy as np 
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon
import matplotlib.pyplot as plt 


## Load the Shepp Logan Phantom
img = shepp_logan_phantom()

def radonTransform(image, theta_steps):
    theta = np.linspace(0,180, theta_steps)
    sinogram = radon(image, theta)
    reconstruction = iradon(sinogram, theta)
    return sinogram, reconstruction

## Test it with different numbers of steps
sino1, recon1 = radonTransform(img, 400)
sino2, recon2 = radonTransform(img, 200)
sino3, recon3 = radonTransform(img, 50)
sino4, recon4 = radonTransform(img, 25)

## Plot the reconstructions
fig1, ax1 = plt.subplots(2,2)
ax1[0,0].imshow(recon1, cmap="gray")
ax1[0,1].imshow(recon2, cmap="gray")
ax1[1,0].imshow(recon3, cmap="gray")
ax1[1,1].imshow(recon4, cmap="gray")
ax1[0,0].set_title(u"$\\theta$ steps = 400")
ax1[0,1].set_title(u"$\\theta$ steps = 200")
ax1[1,0].set_title(u"$\\theta$ steps = 50")
ax1[1,1].set_title(u"$\\theta$ steps = 25")

for a in ax1.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()


'''
Using lower values for n (or fewer scan angles) results in images with aliasing artifacts, 
resulting in lower image quality in the reconstruction.  This means in the context of CT imaging,
using fewer scan angles will result in "worse" images.  However, using more scan angles to 
improve image quality also increases the radiation dose for the patient.  This means the image
quality using this method is inversely related to the radiation the patient endures.
'''