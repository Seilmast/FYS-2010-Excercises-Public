import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from PIL import Image

'''
Part 1.a
'''
## Define the gaussian function and x values to use
g = lambda sig,x: 1/(sig*np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sig**2))
N = 2**8
xlim = 0.5
x = np.linspace(-xlim,xlim, N)
f = np.fft.fftshift( np.fft.fftfreq(len(x), x[1]-x[0]) )


## Compute and plot the gaussians and their Fourier transforms
fig1a,ax1a = plt.subplots(1,2, figsize=(10,5))
sigvec = [0.1, 0.02, 0.01]
for sig in sigvec:
    y = g(sig,x)
    Y = np.fft.fftshift( np.fft.fft(y) )

    lab = f"$\sigma={sig}$"
    ax1a[0].plot(x, y, label=lab)
    ax1a[1].plot(f, np.abs(Y), label=lab)

ax1a[0].set_xlabel("$x$")
ax1a[0].set_ylabel("$g_\sigma(x)$")
ax1a[1].set_xlabel("$f$")
ax1a[1].set_ylabel("$G_\sigma(f)$")
ax1a[0].legend()
ax1a[1].legend()
fig1a.suptitle("Part 1.a")
plt.tight_layout()


'''
Part 1.b
'''
h = g(0.01, x) - g(0.02, x)
H = np.fft.fftshift( np.fft.fft(h) )

fig1b,ax1b = plt.subplots(1,2, figsize=(10,5))
ax1b[0].plot(x, h, label="$g_{{0.01}}(x) - g_{{0.02}}(x)$")
ax1b[1].plot(f, np.abs(H), label="$G_{{0.01}}(x) - G_{{0.02}}(x)$")
ax1b[0].legend()
ax1b[1].legend()
ax1b[0].set_xlabel("$x$")
ax1b[0].set_ylabel("$h$")
ax1b[1].set_xlabel("$f$")
ax1b[1].set_ylabel("$H$")
fig1b.suptitle("Part 1.b")
plt.tight_layout()



'''
Part 2.a
'''
N = 2**5
_f = np.linspace(-5,5,N)
# _x = 

## A gaussian centered at zero in the frequency domain is a
# low-pass filter.
GLP = g(0.1,_f)
GLP /= GLP.max()

## Use the equation given to derive the complementary high-
# pass filter.
GHP = np.sqrt(1 - GLP**2)

## Compute the inverse Fourier transforms
glp = np.fft.ifftshift( np.fft.ifft( np.fft.fftshift(GLP) ) ).real
ghp = np.fft.ifftshift( np.fft.ifft( np.fft.fftshift(GHP) ) ).real

fig2a,ax2a = plt.subplots(1,2, figsize=(10,5))
ax2a[0].plot(_f, GHP, label="$G_{{HP}}$")
ax2a[0].plot(_f, GLP, label="$G_{{LP}}$")
ax2a[1].plot(ghp, label="$g_{{HP}}$")
ax2a[1].plot(glp, label="$g_{{LP}}$")
ax2a[0].set_xlabel("f")
ax2a[0].legend()
ax2a[1].legend()
fig2a.suptitle("Part 2.a")


'''
Part 2.b
'''
## Define the rectangular function f(x)
N = 2**10
x = np.linspace(-3,3, N)
f = np.zeros(len(x))
f[np.abs(x) < 1] = 1

## Convolve f with glp and ghp
flp = np.convolve(f, glp, mode="same")
fhp = np.convolve(f, ghp, mode="same")

fig2b,ax2b = plt.subplots(1,3, figsize=(12,5))
ax2b[0].plot(x, f, label="$f$")
ax2b[1].plot(x, flp, label="$f*g_{{LP}}$")
ax2b[1].plot(x, fhp, label="$f*g_{{HP}}$")
# ax2b[2].plot(x, flp**2 + fhp**2, label="$(f*g_{{LP}})^2 + (f*g_{{HP}})^2$")
ax2b[2].plot(x, flp + fhp, label="$(f*g_{{LP}}) + (f*g_{{HP}})$")

ax2b[0].legend()
ax2b[1].legend()
ax2b[2].legend()
fig2b.suptitle("Part 2.b")



'''
Part 2.c
'''
## Load the image
im = np.array( Image.open("Fig0222(b)(cameraman).tif") )
# im = np.asarray(im)

## Create the 2d gaussian filter in the frequency domain
# Define the 2d frequency space
lims = (-1,1)
N=2**5
x,y = [np.linspace(lims[0],lims[1],N) for _ in range(2)]
X,Y = np.meshgrid(x,y)
def g_2d(sigma, x, y):
    return np.exp( (-1/2)*(x**2/(sigma**2) + y**2/(sigma**2)) )

## Define the lowpass filter
GLP_2d = g_2d(0.1, X,Y)
GLP_2d = GLP_2d / GLP_2d.max()

## Define the highpass filter
GHP_2d = np.sqrt(1 - GLP_2d**2)

## Define the spatial domain filters
glp_2d = np.fft.ifftshift( np.fft.ifft2( np.fft.fftshift(GLP_2d) ) )
ghp_2d = np.fft.ifftshift( np.fft.ifft2( np.fft.fftshift(GHP_2d) ) )
glp_2d = np.real(glp_2d)
ghp_2d = np.real(ghp_2d)

## Compute the convolutions
im_lp = convolve2d(im, glp_2d, "same")
im_hp = convolve2d(im, ghp_2d, "same")

## Plot the images
fig2c,ax2c = plt.subplots(2,5, figsize=(12,5))
ax2c[0,0].imshow(im, cmap="gray")
ax2c[0,1].imshow(GLP_2d)
ax2c[1,1].imshow(GHP_2d)
ax2c[0,2].imshow(glp_2d)
ax2c[1,2].imshow(ghp_2d)
ax2c[0,3].imshow(im_lp, cmap="gray")
ax2c[1,3].imshow(im_hp, cmap="gray")
# ax2c[0,4].imshow( np.sqrt(im_lp**2 + im_hp**2) , cmap="gray")
ax2c[0,4].imshow( im_lp + im_hp , cmap="gray")

ax2c[0,0].set_title("f")
ax2c[0,1].set_title("$G_{{LP}}$")
ax2c[1,1].set_title("$G_{{HP}}$")
ax2c[0,2].set_title("$g_{{LP}}$")
ax2c[1,2].set_title("$g_{{HP}}$")
ax2c[0,3].set_title("$f*g_{{LP}}$")
ax2c[1,3].set_title("$f*g_{{HP}}$")
# ax2c[0,4].set_title("$(f*g_{{LP}})^2 + (f*g_{{HP}})^2$")
ax2c[0,4].set_title("$(f*g_{{LP}}) + (f*g_{{HP}})$")


for a in ax2c.ravel(): a.set_axis_off()
fig2c.suptitle("Part 2.c")

plt.show()
