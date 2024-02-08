import numpy as np
import matplotlib.pyplot as plt

'''
@THIS IS NOT FINISHED
'''

## Define an arbitrary value for a and a vector of i to plot against
a = 10
ivec = np.linspace(-3.5*a,3.5*a,1000)


## Define the ramp function h(i)
def h(i):
    # If the input is a single value
    if isinstance(i, float) or isinstance(i,int):
        return i if 0<=i and i<=a/5 else 0
    # If the input is a vector
    elif isinstance(i, np.ndarray):
        mask = np.logical_and(0<=i, i<=a/5)
        return mask * i

## Define g(i) as a sum of ramp functions
def g(i):
    return 0.5*h(.5*(i+a)) + 0.5*h(-.5*(i+a/5)) + h(i+a/5) + h(-1*(i-a/5)) - 0.5*h(.5*(i-a/5)) - 0.5*h(-.5*(i-a))




'''
Part (a)
'''
## Define f(i)
def f_a(i):
    return np.logical_and(-a<=i, i<=a).astype(float)
    
## Define p(i) = (f*h)(i)
def p_a(i):
    condition_1 = 0.5 * (i + a)**2
    condition_2 = a**2 / 50
    condition_3 = -0.5 * i**2 + a*i - 12/25 * a**2

    mask_1 = np.logical_and(-a <= i, i < -a + a/5)
    mask_2 = np.logical_and(-a + a/5 <= i, i < a)
    mask_3 = np.logical_and(a < i, i <= a + a/5)

    return condition_1*mask_1 + condition_2*mask_2 + condition_3*mask_3


## Plot f_a(i), h(i), and p_a(i)
fig_a1,ax_a1 = plt.subplots(2,1)
ax_a1[0].plot(ivec, f_a(ivec), label="$f_a(i)$")
ax_a1[0].plot(ivec, h(ivec), label="$h(i)$")
ax_a1[0].legend()
ax_a1[1].plot(ivec, p_a(ivec), label="$(f_a*h)(i)$")
ax_a1[1].legend()
plt.legend()


## Compute (f*g)(i) as a sum of transformations of p(i)
def f_a_conv_g(i):
    return 0.5*p_a(.5*(i+a)) + 0.5*p_a(-.5*(i+a/5)) + p_a(i+a/5) + p_a(-1*(i-a/5)) - 0.5*p_a(.5*(i-a/5)) - 0.5*p_a(-.5*(i-a))


## Plot f(i), g(i) and (f*g)(i)
fig_a2, ax_a2 = plt.subplots(3,1)
ax_a2[0].plot(ivec, f_a(ivec), label="$f_a(i)$")
ax_a2[1].plot(ivec, g(ivec), label="$g(i)$")
ax_a2[2].plot(ivec, f_a_conv_g(ivec), label="$(f_a*g)(i)$")
ax_a2[0].legend()
ax_a2[1].legend()
ax_a2[2].legend()
plt.tight_layout()
plt.show()




'''
Part (b)
'''
## Define f(i)
def f_b(i):
    return np.logical_and(-a/2<=i, i<=a/2).astype(float) * 2

## Define p(i) = (f*h)(i)
def p_b(i):
    condition_1 = i**2 + a*i + a**2/4
    condition_2 = a**2 / 25
    condition_3 = -i**2 + a*i - 21/100*(a**2)
    # condition_3 = a*(i - 23*a/50) - i**2

    mask_1 = np.logical_and(-a/2 <= i, i < -a/2 + a/5)
    mask_2 = np.logical_and(-a/2 + a/5 <= i, i < a/2)
    mask_3 = np.logical_and(a/2 < i, i <= a/2 + a/5)

    return condition_1*mask_1 + condition_2*mask_2 + condition_3*mask_3
    
    
## Plot f_b(i), h(i), and p_b(i)
fig_b1,ax_b1 = plt.subplots(2,1)
ax_b1[0].plot(ivec, f_b(ivec), label="$f_b(i)$")
ax_b1[0].plot(ivec, h(ivec), label="$h(i)$")
ax_b1[0].legend()
ax_b1[1].plot(ivec, p_b(ivec), label="$(f_b*h)(i)$")
ax_b1[1].legend()
plt.legend()


## Compute (f*g)(i) as a sum of transformations of p(i)
def f_b_conv_g(i):
    return 0.5*p_b(.5*(i+a)) + 0.5*p_b(-.5*(i+a/5)) + p_b(i+a/5) + p_b(-1*(i-a/5)) - 0.5*p_b(.5*(i-a/5)) - 0.5*p_b(-.5*(i-a))


## Plot f(i), g(i) and (f*g)(i)
fig_b2, ax_b2 = plt.subplots(3,1)
ax_b2[0].plot(ivec, f_b(ivec), label="$f_b(i)$")
ax_b2[1].plot(ivec, g(ivec), label="$g(i)$")
ax_b2[2].plot(ivec, f_b_conv_g(ivec), label="$(f_b*g)(i)$")
ax_b2[0].legend()
ax_b2[1].legend()
ax_b2[2].legend()
plt.tight_layout()
plt.show()
