import os
import shutil
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from skimage import transform
from scipy import special

for bird in os.listdir("../real_data/consolidated_dataset/"):
    if bird == ".DS_Store":
        continue
    for image in os.listdir("../real_data/consolidated_dataset/"+bird):
        if image == ".DS_Store":
            continue
        im = io.imread("../real_data/consolidated_dataset/"+bird+"/"+image)
        im = transform.resize(im, (224, 224))
        draw = np.random.uniform(low = 0, high = 1, size = 1)
        # Put in testing dataset
        if draw < 0.15:
            io.imsave("./../real_data/final_data/test/"+bird+"/"+image, im)
        # Put in validation dataset
        elif draw >=0.15 and draw < 0.3:
            io.imsave("./../real_data/final_data/val/"+bird+"/"+image, im)
        # Put in training dataset
        elif draw >= 0.3:
            io.imsave("./../real_data/final_data/train/"+bird+"/"+image, im)
