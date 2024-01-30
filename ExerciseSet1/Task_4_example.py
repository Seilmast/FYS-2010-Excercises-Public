import numpy as np 
import matplotlib.pyplot as plt 


width = 10

#Make a gradient image where each value has "width" columns 
gradient = np.zeros((300, 256*width))

for col in range(gradient.shape[1]):
    gradient[:,col] = col//10

fig, ax = plt.subplots(4,2,figsize=(20,10))
ax[0,0].imshow(gradient, cmap='gray')
ax[0,0].title.set_text('8-bit')

#Here we do col//20 as two values must share the same digitization to simulate a 7-bit system
#This logic holds for all future for loops going down
for col in range(gradient.shape[1]):
    gradient[:,col] = col//20

ax[0,1].imshow(gradient, cmap='gray')
ax[0,1].title.set_text('7-bit')

for col in range(gradient.shape[1]):
    gradient[:,col] = col//40

ax[1,0].imshow(gradient, cmap='gray')
ax[1,0].title.set_text('6-bit')

for col in range(gradient.shape[1]):
    gradient[:,col] = col//80

ax[1,1].imshow(gradient, cmap='gray')
ax[1,1].title.set_text('5-bit')

for col in range(gradient.shape[1]):
    gradient[:,col] = col//160

ax[2,0].imshow(gradient, cmap='gray')
ax[2,0].title.set_text('4-bit')

for col in range(gradient.shape[1]):
    gradient[:,col] = col//320

ax[2,1].imshow(gradient, cmap='gray')
ax[2,1].title.set_text('3-bit')

for col in range(gradient.shape[1]):
    gradient[:,col] = col//640

ax[3,0].imshow(gradient, cmap='gray')
ax[3,0].title.set_text('2-bit')

for col in range(gradient.shape[1]):
    gradient[:,col] = col//1280

ax[3,1].imshow(gradient, cmap='gray')
ax[3,1].title.set_text('1-bit')


plt.tight_layout()

plt.show()
