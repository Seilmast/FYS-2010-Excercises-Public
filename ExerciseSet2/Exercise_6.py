import numpy as np
import matplotlib.pyplot as plt 

x = np.linspace(0, 2, num=100)

def task_a(x, A, L0):
    return A*np.exp(-(np.log(2)*np.power(x, 2))/(L0**2))

def task_b(x, B, L0):
    return B*(1 - np.exp(-(np.log(2)*np.power(x, 2))/(L0**2)))

def task_c(x, C, D, alpha):
    return (D-C)*(1 - np.exp(-(alpha*np.power(x, 2)))) + C


fig, ax = plt.subplots(1,3, figsize=(12,4), sharex=True, sharey=True)
ax[0].plot(x, task_a(x, 1, 0.5))
ax[0].set_title('Subtask a')

ax[1].plot(x, task_b(x, 1, 0.5))
ax[1].set_title('Subtask b')

ax[2].plot(x, task_c(x, 0.5, 1, 1))
ax[2].set_title('Subtask a')


plt.tight_layout()
plt.show()