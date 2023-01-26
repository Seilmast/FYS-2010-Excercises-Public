from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



def T1(r, A, L0):
	return A*np.exp((-r**2 * np.log(2)) * L0**-2)


def T2(r, B, L0):
	return B - T1(r,B,L0)

def T3(r, C, D):
	return D - C*np.exp(-r**2)


x = np.linspace(0,6,500)

a = 2
l0=2
y = T1(x, A=a, L0=l0)
print( T1(0, A=a, L0=l0), T1(l0, A=a, L0=l0))
plt.plot(x, y)

b = 2
l0 = 2
y = T2(x, B=b, L0=l0)
print( T2(0, B=b, L0=l0), T2(l0, B=b, L0=l0))
plt.plot(x, y)

c = 1
d = 2
y = T3(x, C=c, D=d)
print( T3(0, C=c, D=d), T3(1000, C=c, D=d))
plt.plot(x, y)

plt.legend(["T1","T2"])
plt.show()