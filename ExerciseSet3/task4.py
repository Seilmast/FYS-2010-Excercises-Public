import numpy as np
import matplotlib.pyplot as plt

'''
@THIS IS NOT FINISHED
'''


'''
Part (a)
'''
## Define g(i)
a = 10
nsteps = 10000
i_vec = np.linspace(-2*a,2*a, nsteps)
delta = i_vec[1] - i_vec[0]
g = np.zeros(i_vec.shape)

for ix,i in enumerate(i_vec):
    if i >= -a and i < -3*a/5:
        g[ix] = (5/(2*a))*i + 2.5
    elif i >= -3*a/5 and i < -a/5:
        g[ix] = -(5/(2*a))*i - 0.5
    elif i >= -a/5 and i < 0:
        g[ix] = (2*5/a)*i + 2
    elif i >= 0 and i < a/5:
        g[ix] = -(2*5/a)*i + 2
    elif i >= a/5 and i < 3*a/5:
        g[ix] = -(5/(2*a))*i + 0.5
    elif i >= 3*a/5 and i < a:
        g[ix] = (5/(2*a))*i - 2.5

## Define f(i)
f = np.zeros(i_vec.shape)
f[np.logical_and(i_vec>=-a, i_vec<a)] = 1
i_vec2 = i_vec[f!=0]
f = f[f!=0]

## Convolve (g*f)(i)
# conv = np.convolve(g,f)[500:1500]
conv = np.convolve(g,f, mode="full") * delta
i_vec3 = np.arange(i_vec[0]+i_vec2[0], i_vec[-1]+i_vec2[-1], step=delta)
print(i_vec3.min(), i_vec.min(), i_vec2.min())
print(len(i_vec3), len(i_vec), len(i_vec2))
print(len(conv), len(g), len(f))

plt.plot(i_vec, g, label="g(i)")
plt.plot(i_vec2, f, label="f(i)")
plt.plot(i_vec3, conv, label="(g*f)(x)")
# plt.yticks([-1,0,1,2])
# plt.xticks(a * np.linspace(-2,2,11))
plt.legend()
plt.show()