import numpy as np
from matplotlib import pyplot as plt

'''
Part 1.a
'''
## Define the gaussian function and x values to use
g = lambda sig,x: 1/(sig*np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sig**2))
N = 2**8
xlim = -1
x = np.linspace(-xlim,xlim, N)
f = np.fft.fftshift( np.fft.fftfreq(len(x)) )

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
ax1b[0].plot(x, h)
ax1b[1].plot(f, np.abs(H))
ax1b[0].set_xlabel("$x$")
ax1b[0].set_ylabel("$h$")
ax1b[1].set_xlabel("$f$")
ax1b[1].set_ylabel("$H$")
fig1b.suptitle("Part 1.b")
plt.tight_layout()



'''
Part 2.a
'''
_f = np.linspace(-0.5,0.5,N)
# _x = 

## A gaussian centered at zero in the frequency domain is a
# low-pass filter.
GLP = g(0.1,_f)
GLP /= GLP.max()

## Use the equation given to derive the complementary high-
# pass filter.
GHP = np.sqrt(1 - GLP**2)

## Compute the inverse Fourier transforms
glp = np.fft.ifftshift( np.fft.ifft( GLP ) )
ghp = np.fft.ifftshift( np.fft.ifft( GHP ) )
# glp = np.abs(glp)
# ghp = np.abs(ghp)

fig2a,ax2a = plt.subplots(1,2, figsize=(10,5))
ax2a[0].plot(GLP, label="$G_{{LP}}$")
ax2a[0].plot(GHP, label="$G_{{HP}}$")
ax2a[1].plot(np.real(glp), label="$g_{{LP}}$")
ax2a[1].plot(np.real(ghp), label="$g_{{HP}}$")
ax2a[0].legend()
ax2a[1].legend()
fig2a.suptitle("Part 2.a")


'''
Part 2.b
'''
## Define the rectangular function f(x)
N = 2**8
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
ax2b[2].plot(x, flp**2 + fhp**2)
# ax2b[0].legend()
ax2b[1].legend()
fig2b.suptitle("Part 2.b")


plt.show()

