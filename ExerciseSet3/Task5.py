import numpy as np 
import matplotlib.pyplot as plt 


'''
Part (a)
'''
## Define the vector
f = np.array([0,0,0,1,1,1,1,1,1,1,1,1,0,0,0])

## Find the FFT of the vector and fftshift it
F = np.fft.fft(f)
F_shift = np.fft.fftshift(F)

## Find the phase and magnitude of the vector with and without
#  fftshift
phi = np.angle(F)
mag = np.absolute(F)

phi_shift = np.angle(F_shift)
mag_shift = np.absolute(F_shift)


## Plot the shifted and unshifted values
fig,ax = plt.subplots(2,2, figsize=(6,6))
ax[0,0].stem(mag)
ax[0,0].set_title("magnitude without shift")
ax[1,0].stem(phi)
ax[1,0].set_title("phase without shift")
ax[0,1].stem(mag_shift)
ax[0,1].set_title("magnitude with shift")
ax[1,1].stem(phi_shift)
ax[1,1].set_title("phase with shift")

fig.suptitle("Without zero-padding")
plt.tight_layout()


'''
Shifting the FFT of the signal centers the signal in the frequency window.

Equation (4-76) displays the relationship between frequency shifting like this
in the time domain and the frequency domain for 2-dimensional signals.  This
problem uses a 1-dimensional signal.
'''


'''
Part (b)
'''
zeros = np.zeros(8)
f_pad = np.concatenate([zeros,f,zeros])
print(f)
print(f_pad)

## Recompute the FFT with and without shifting
F_pad = np.fft.fft(f_pad)
F_shift_pad = np.fft.fftshift(F_pad)

## Find the phase and magnitude of the vector with and without
#  fftshift
phi_pad = np.angle(F_pad)
mag_pad = np.absolute(F_pad)

phi_shift_pad = np.angle(F_shift_pad)
mag_shift_pad = np.absolute(F_shift_pad)

## Plot the shifted and unshifted values
fig,ax = plt.subplots(2,2, figsize=(6,6))
ax[0,0].stem(mag_pad)
ax[0,0].set_title("magnitude without shift")
ax[1,0].stem(phi_pad)
ax[1,0].set_title("phase without shift")
ax[0,1].stem(mag_shift_pad)
ax[0,1].set_title("magnitude with shift")
ax[1,1].stem(phi_shift_pad)
ax[1,1].set_title("phase with shift")

fig.suptitle("With zero-padding")
plt.tight_layout()

plt.show()

'''
Zero-padding the signal increases the "resolution" of the FFT.  The 
overall shape of the signals is similar, but there are more samples
describing the signal in the frequency domain.
'''
