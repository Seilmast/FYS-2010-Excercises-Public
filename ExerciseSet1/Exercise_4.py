import numpy as np
import matplotlib.pyplot as plt

imsize = (2, 2)

K = 255
x0, y0 = imsize[0]/2,imsize[1]/2
x, y = np.linspace(0, imsize[0], num=1000), np.linspace(0, imsize[1], num=1000)
X, Y = np.meshgrid(x, y)
Z = K*np.exp(-((X-x0)**2 + (Y-y0)**2))

plt.figure()
plt.contourf(X,Y,Z)
plt.show()