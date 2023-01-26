import numpy as np
import matplotlib.pyplot as plt 


def task_a(x, A, L0):
    return A*np.exp(-(np.log(2)*np.power(x, 2))/(L0**2))

def task_b(x, B, L0):
    return B*(1 - np.exp(-(np.log(2)*np.power(x, 2))/(L0**2)))

def task_c(x, C, D, alpha):
    return (D-C)*(1 - np.exp(-(alpha*np.power(x, 2)))) + C

x = np.linspace(0, 2, num=100)

'''
Subtask a
'''
L0 = 0.5
A = 1
fig, ax = plt.subplots(1,3, figsize=(12,4), sharex=True, sharey=True)
ax[0].plot(x, task_a(x, A, L0))
ax[0].scatter([L0], [A/2], c="r")
ax[0].set_title('Subtask a')

'''
Subtask b
'''
L0 = 0.5
B = 1
ax[1].plot(x, task_b(x, B, L0))
ax[1].scatter([L0], [B/2], c="r")
ax[1].plot(x, np.full(x.shape, B), "--r")
ax[1].set_title('Subtask b')

'''
Subtask c
'''
C = 0.2
D = 1
alpha = 1
ax[2].plot(x, task_c(x, C, D, alpha))
ax[2].plot(x, np.full(x.shape, C), "--r")
ax[2].plot(x, np.full(x.shape, D), "--r")
ax[2].set_title('Subtask c')


plt.tight_layout()
plt.show()