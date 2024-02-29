import numpy as np
import matplotlib.pyplot as plt


'''
6.1
'''
## Define the "warm white" image
rgb = np.array([[0.90, 0.85, 0.50]]) 
img = np.ones((100,100,3)) * 255 * rgb
img = img.astype(np.uint8)

## Plot the color
fig = plt.figure()
ax = fig.add_axes(111)
ax.imshow(img)
ax.set_title(f'"warm white": {rgb}')
ax.set_axis_off()



'''
6.5

At the center of the image, the RGB value will be [0.5, 1, 0.5].  This color
vector corresponds to a light green.
'''
## Make the image
N = 100
red = np.linspace(1,0,N)
green = np.concatenate([np.linspace(0,1,N//2), np.linspace(1,0,N//2)])
blue = np.linspace(0,1,N)

img = np.stack([red, green, blue], axis=1)
img = np.tile(img, reps=(N,1,1))

## Plot the graphs and image
fig = plt.figure(figsize=(10,5))

ax_graphs = fig.add_axes(121)
ax_graphs.plot(red, c="r")
ax_graphs.plot(blue, c="b")
ax_graphs.plot(green, c="g")

ax_img = fig.add_axes(122)
ax_img.imshow(img)
ax_img.set_axis_off()




'''
6.8
'''
n_steps = 50
vec = np.linspace(0,1,n_steps)

fig,ax = plt.subplots(3,3, figsize=(7,7))

## The top face of the color cube is where B=1
blue_top = np.ones((n_steps,n_steps))
red_top = np.tile(vec, (n_steps,1))
green_top = np.tile(vec[::-1], (n_steps,1))
green_top = np.rot90(green_top)
top = np.stack([red_top, green_top, blue_top], axis=2)

ax[0,0].imshow(red_top, cmap="gray", vmin=0, vmax=1)
ax[1,0].imshow(green_top, cmap="gray", vmin=0, vmax=1)
ax[2,0].imshow(blue_top, cmap="gray", vmin=0, vmax=1)
ax[0,0].set_title("Top")


## The front face is where R=1
red_front = np.ones((n_steps,n_steps))
blue_front = np.tile(vec, (n_steps,1))
blue_front = np.rot90(blue_front)
green_front = np.tile(vec, (n_steps,1))
green_front = np.rot90(green_top)
front = np.stack([red_top, green_top, blue_top], axis=2)

ax[0,1].imshow(red_front, cmap="gray", vmin=0, vmax=1)
ax[1,1].imshow(green_front, cmap="gray", vmin=0, vmax=1)
ax[2,1].imshow(blue_front, cmap="gray", vmin=0, vmax=1)
ax[0,1].set_title("Front")


## The right face is where G=1
green_right = np.ones((n_steps,n_steps))
blue_right = np.tile(vec, (n_steps,1))
blue_right = np.rot90(blue_right)
red_right = np.tile(vec[::-1], (n_steps,1))
right = np.stack([red_top, green_top, blue_top], axis=2)

ax[0,2].imshow(red_right, cmap="gray", vmin=0, vmax=1)
ax[1,2].imshow(green_right, cmap="gray", vmin=0, vmax=1)
ax[2,2].imshow(blue_right, cmap="gray", vmin=0, vmax=1)
ax[0,2].set_title("Right")

## Label the colors
ax[0,0].set_ylabel("R")
ax[1,0].set_ylabel("G")
ax[2,0].set_ylabel("B")

# for a in ax.ravel():
#     a.set_axis_off()

plt.show()
