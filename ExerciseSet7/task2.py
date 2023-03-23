import numpy as np
import matplotlib.pyplot as plt



'''
Define and plot the Gabor Wavelet and the Scaled Gabor Wavelet
'''
#### Define the Gabor wavelet function and its scaled variant
gabor = lambda sig,x: 1/(sig * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sig**2)) * np.cos(2*np.pi*x)
scaled_gabor = lambda sig,s,x: 1/np.sqrt(s) * gabor(sig, x/s)

#### Plot the wavelet variations
N = 2**8
xvec = np.linspace(-5,5, N)
fs = (xvec.max() - xvec.min())/N
frq_vec = (np.arange(0,N) - N//2) / N * fs

# xvec = np.linspace(-4,4, N)
w0 = scaled_gabor(1,0.5,xvec)
w1 = scaled_gabor(2,0.5,xvec)
w2 = scaled_gabor(1,1,xvec)

fig1,ax1 = plt.subplots(1,1)
ax1.plot(xvec, w0, label="$\sigma=1, s=0.5$")
ax1.plot(xvec, w1, label="$\sigma=2, s=0.5$")
ax1.plot(xvec, w2, label="$\sigma=1, s=1$")
ax1.set_ylabel("$w_{\sigma,s}$")
ax1.set_xlabel("x")
plt.legend()
# plt.show()

'''
Compute the Fourier transforms of several wavelets.
'''
W0 = np.fft.fftshift( np.fft.fft(w0) )
W1 = np.fft.fftshift( np.fft.fft(w1) )
W2 = np.fft.fftshift( np.fft.fft(w2) )

fig2, ax2 = plt.subplots(1,1)
ax2.plot(frq_vec, np.absolute(W0), label="$\sigma=1, s=0.5$")
ax2.plot(frq_vec, np.absolute(W1), label="$\sigma=2, s=0.5$")
ax2.plot(frq_vec, np.absolute(W2), label="$\sigma=1, s=1$")
ax2.set_ylabel("dB")
ax2.set_xlabel("$f$")
plt.legend()



'''
Construct the signal S
'''
## Plotting values
N = 2**10-1
xvec = np.linspace(-4,4, N)
fs = (xvec.max() - xvec.min())/N
frq_vec = (np.arange(0,N) - N//2) / N * fs

## Compute and add the FFT of each wavelet
expvec = np.linspace(-5,1, num=7)
S = np.zeros(N)
for exp in expvec:
    w = scaled_gabor(0.5, 2**exp, xvec)
    W = np.fft.fftshift( np.fft.fft(w) )
    S += np.absolute(W)


fig3,ax3 = plt.subplots(1,1)
ax3.plot(frq_vec, S, label="S")
ax3.set_ylabel("S")
ax3.set_xlabel("$f$")
plt.legend()

plt.show()

'''
The signal S has a sharp valley at frequency 0.  Using this set of wavelets
to decompose a signal would lose information at low frequencies.  To avoid this,
low-pass filters should be added.
'''