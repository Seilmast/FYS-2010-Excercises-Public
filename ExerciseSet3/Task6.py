import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

## Load the images
x1 = Image.open("Data/Fig0424(a)(rectangle).tif")
x2 = Image.open("Data/Fig0425(a)(translated_rectangle).tif")
x3 = x1.rotate(45)
x4 = x2.rotate(45)

## Take the fft of the images
f1 = np.fft.fftshift( np.fft.fft2(x1) )
f2 = np.fft.fftshift( np.fft.fft2(x2) )
f3 = np.fft.fftshift( np.fft.fft2(x3) )
f4 = np.fft.fftshift( np.fft.fft2(x4) )

## Plot the log-intensities of the phase and magnitude
fig,ax = plt.subplots(4,3, figsize=(5,7))
ax[0,0].imshow( x1, cmap="gray" )
ax[0,1].imshow( np.log(np.absolute(f1)), cmap="gray" )
ax[0,2].imshow( np.angle(f1), cmap="gray" )

ax[1,0].imshow( x2, cmap="gray" )
ax[1,1].imshow( np.log(np.absolute(f2)), cmap="gray" )
ax[1,2].imshow( np.angle(f2), cmap="gray" )

ax[2,0].imshow( x3, cmap="gray" )
ax[2,1].imshow( np.log(np.absolute(f3)), cmap="gray" )
ax[2,2].imshow( np.angle(f3), cmap="gray" )

ax[3,0].imshow( x4, cmap="gray" )
ax[3,1].imshow( np.log(np.absolute(f4)), cmap="gray" )
ax[3,2].imshow( np.angle(f4), cmap="gray" )

ax[0,0].set_title("Original Image")
ax[0,1].set_title("Magnitude of FFT")
ax[0,2].set_title("Phase of FFT")
for a in ax.ravel(): a.set_axis_off()
plt.tight_layout()

plt.show()

