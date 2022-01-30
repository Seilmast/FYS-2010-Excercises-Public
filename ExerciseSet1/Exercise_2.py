from PIL import Image
import matplotlib.pyplot as plt 

###Subtask a###

img_path = 'ExerciseSet1\Data\Fig0227(a)(washington_infrared).tif'
img = Image.open(img_path)

plt.figure()
plt.imshow(img, cmap='gray')
plt.show()