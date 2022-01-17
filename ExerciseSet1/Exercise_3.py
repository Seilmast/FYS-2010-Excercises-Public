from random import random
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np

###Subtask a###

img_path = 'ExerciseSet1\Data\Fig0207(a)(gray level band).tif'
image = Image.open(img_path)

#To make the image easier to work with, we'll convert it no ndarray
img = np.asarray(image)

#Lets choose a random row to plot the intensities of
random_row = np.random.randint(0, len(img))
intensities = img[random_row, :]
x = np.linspace(0, len(intensities), num = len(intensities))

fig, ax = plt.subplots(1,2, figsize=(10,2))
ax[0].imshow(img, cmap='gray', aspect="auto")
ax[0].set_title('Image')
ax[1].plot(intensities)
ax[1].set_title(f"Intensities on row {random_row}")


plt.tight_layout()
plt.show()
plt.close()

print(f"There are {len(np.unique(intensities))} unique intensities")

###Subtask b###
print(f"The shape of img is {img.shape}")


###Subtask c###
unique_band_values = np.unique(img)
#Stringcodes for the different colors for the bands 
print(unique_band_values)
c = ['green', 'green', 'red', 'orange', 'yellow', 'purple', 'black']

fig, ay = plt.subplots(1,2, figsize=(10,2))
ay[0].imshow(img, cmap='gray', aspect="auto")
ay[0].set_title('Greyscale image')
ay[1].contourf(img, levels = unique_band_values, colors=c)
ay[1].set_title(f"Color image")

for a in ay.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
plt.close()