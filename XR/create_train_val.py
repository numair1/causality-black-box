import os
import numpy as np
import shutil
np.random.seed(0)

for img in os.listdir("./data_img/normal"):
    if np.random.uniform(0, 1) <= 0.75:
        shutil.copy("./data_img/normal/"+img, "./data_train_val/train/normal/"+img)
    else:
        shutil.copy("./data_img/normal/"+img, "./data_train_val/val/normal/"+img)

for img in os.listdir("./data_img/pneumonia"):
    if np.random.uniform(0, 1) <= 0.75:
        shutil.copy("./data_img/pneumonia/"+img, "./data_train_val/train/pneumonia/"+img)
    else:
        shutil.copy("./data_img/pneumonia/"+img, "./data_train_val/val/pneumonia/"+img)
