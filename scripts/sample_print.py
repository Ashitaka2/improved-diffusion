import numpy as np
import matplotlib.pyplot as plt

# data = np.load('train/samples_1024x32x32x3.npz')['arr_0'] 
data = np.load('samples3/samples_1024x32x32x3.npz')['arr_0'] 

# Plot the first 100 images
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i, ax in enumerate(axes.flat):
    ax.imshow(data[i])
    ax.axis('off')

plt.savefig('samples3/naive_no_time.png')
# plt.show()