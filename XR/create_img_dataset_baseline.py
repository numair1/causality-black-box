import pandas as pd
import numpy as np
import os
import shutil
np.random.seed(0)

img_csv = pd.read_csv("./XR/Data_Entry_2017_v2020.csv")

img_id = img_csv.loc[img_csv['Finding Labels'] == 'No Finding']['Image Index'].values

img_id_sample = np.random.choice(img_id, size = 120)

for im in img_id_sample:
    print(im in os.listdir("./data_img/pneumonia"))
    if np.random.uniform(0,1)<= 0.75:
        shutil.copy("./XR/data_files/images/"+im, "./data_alternate_baseline/train/normal/"+im)
    else:
        shutil.copy("./XR/data_files/images/"+im, "./data_alternate_baseline/val/normal/"+im)
