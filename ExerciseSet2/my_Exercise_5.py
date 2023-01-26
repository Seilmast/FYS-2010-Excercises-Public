from PIL import Image
import matplotlib.pyplot as plt
import numpy as np




def scaleI(im, L):

	A = np.array([[im.min(), 1], [im.max(), 1]])
	y = np.array([[0],[L-1]])
	x = np.linalg.inv(A) @ y

	scaled_im = im*x[0] + x[1]

	return scaled_im



test_im = np.arange(20) + 100

print(test_im.min(), test_im.max())

test_out = scaleI(test_im, 20)

print(test_out.min(), test_out.max())