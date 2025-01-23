import numpy as np
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
###Subtask a###
A = np.zeros((7,8))

print(f'This matrix is a matrix of only zeros, and the shape 7x8:\n{A}\n')

###Subtask b###

#Tips: A quick way to select either an entire row, column, or dimension in general
#is to use semicolon to select the entire dimension. E.g. A[:,0] will select column 0 on all rows and return them as an array, or A[0,:] will select all columns on row 0

A[:,0] = 1 #Give the first column the value 1
A[:,1] = 2 #Give the second column the value 2
A[4,6] = 5 #Give position (5,7) the value 5 (Remember in python indexes start at zero)
A[5,6] = 8 #Give position (6,7) the value 5 (Remember in python indexes start at zero)

print(f'Matrix A has gotten some new values:\n{A}\n')

###Subtask c##
unique_values_A = np.unique(A)
print(f'The unique values of matrix A is {unique_values_A}\n')

###Subtask d###
B = np.random.randint(0, 10, size=(8,7)) #Create a random matrix of size 8x7 with values between 0 and 10. 
print(f'Matrix B has been generated randombly with the shape 8x7:\n{B}\n')

#Tips: If you're uncertain in you matrix's shape, use the numpy .shape feature to find out. 
print(f"The shape of matrix A is: {A.shape}")
print(f"The shape of matrix B is: {B.shape}")


###Subtask e###
#There are a few ways to perform a matrix multiplication in python. 
#The simplest however is to use the @ operator 
C = A@B
print(f"The shape of matrix C is: {C.shape}")

###Subtask f###
print(f'Matrix A looks like this:\n{A}\n')

A_flipped = np.flip(A, axis=1)
print(f'A can be flipped upside down to look like this:\n{A_flipped}\n')

A_flipped = np.flip(A_flipped, axis=0)
print(f'We can also flip the same matrix from left to right as such:\n{A_flipped}\n')

A_rotated = np.rot90(A, k=1, axes=(0, 1))
print(f'We can also manipulate  A by rotating the matrix by 90 degrees as such:\n{A_rotated}')
#This is equal to transposing matrix A, which could be done as easy as writing A.T

#Lastly we can rotate an image to any degrees with the scipy function rotate
A_rotated = rotate(A, 37, axes=(0,1))
print(f'Rotating A by 37 degrees gives us a new matrix with shape {A.shape}, and values:\n{A_rotated}')

'''
So what does rotating the matrix mean? Lets first look at how it changes the matrix
The code below is not needed for the exercise, but will hopefully give some insight into what is going on.
'''

fig, ax = plt.subplots(1,2, figsize=(10,10), sharex=True, sharey=True)
ax[0].matshow(A)
ax[0].set_title('Matrix A')
ax[1].matshow(A_rotated)
ax[1].set_title("Rotated Matrix A")

for a in ax.ravel():
    a.axis('off')

plt.tight_layout()
plt.show()
plt.close()

'''
As you can hopefully see, the matrix has actually rotated. 
If you've had the linear algebra course, you might remember the rotation matrix. 
This has been applies to the image, and is has rotated along some axis.  
'''

###Subtask g###
#We'll save the 37 degree matrix to the folder misc

np.savetxt('misc/rotated_matrix.csv', A_rotated, delimiter=',')