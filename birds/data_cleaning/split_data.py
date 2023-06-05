import os
import numpy as np
from skimage import io
from skimage import transform
from PIL import Image

for bird in os.listdir("../data/consolidated_dataset/"):
    if bird == ".DS_Store":
        continue
    for image in os.listdir("../data/consolidated_dataset/"+bird):
        if image == ".DS_Store":
            continue
        im = io.imread("../data/consolidated_dataset/"+bird+"/"+image)
        im = transform.resize(im, (224, 224))
        im = Image.fromarray((im * 255).astype(np.uint8))
        draw = np.random.uniform(low = 0, high = 1, size = 1)
        # Put in testing dataset
        if draw < 0.15:
            im.save("./../data/final_data/test/"+bird+"/"+image)
        # Put in validation dataset
        elif draw >=0.15 and draw < 0.3:
            im.save("./../data/final_data/val/"+bird+"/"+image)
        # Put in training dataset
        elif draw >= 0.3:
            im.save("./../data/final_data/train/"+bird+"/"+image)
