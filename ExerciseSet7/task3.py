import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import pywt


'''
Part 1
'''
print("\nTypes of wavelets:")
print(pywt.wavelist(kind="discrete"))

img = pywt.data.camera()
cA, (cH,cV,cD) = pywt.dwt2(img, wavelet="sym2")


fig,ax = plt.subplots(2,2)
ax[0,0].imshow(cA, cmap='gray')
ax[0,1].imshow(cH, cmap='gray')
ax[1,0].imshow(cV, cmap='gray')
ax[1,1].imshow(cD, cmap='gray')

ax[0,0].set_title("Approximation")
ax[0,1].set_title("Horizontal")
ax[1,0].set_title("Vertical")
ax[1,1].set_title("Diagonal")

for a in ax.ravel(): a.set_axis_off()
plt.show()



'''
Part 2
'''

