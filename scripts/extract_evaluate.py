import numpy as np
from PIL import Image
import os

data = np.load('samples5/samples_50000x32x32x3.npz')
images = data['arr_0']

output_dir = 'samples5/pretrained_extract'
os.makedirs(output_dir, exist_ok=True)

# print(img.min(), img.max()) #check the pixel range


for idx, img in enumerate(images):
    # Convert the numpy array to a PIL image
#     image = Image.fromarray((img * 255).astype(np.uint8))
    image = Image.fromarray((img).astype(np.uint8))
    image.save(os.path.join(output_dir, f'image_{idx}.png'))

print("extraction complete")