import pandas as pd
import numpy as np
import os
np.random.seed(0)

bbox_csv = pd.read_csv("./BBox_List_2017.csv")

bbox_pneumonia = bbox_csv.loc[bbox_csv['Finding Label'] == "Pneumonia",]
bbox_normal = bbox_csv.loc[bbox_csv['Finding Label'] != "Pneumonia",]

intersection_indices = np.intersect1d(bbox_pneumonia['Image Index'].values, bbox_normal['Image Index'].values)

bbox_normal = bbox_normal.loc[~bbox_normal['Image Index'].isin(intersection_indices)]

bbox_pneumonia_id = bbox_pneumonia['Image Index'].values
bbox_normal_id = np.random.choice(bbox_normal['Image Index'].values, size = 120, replace = False)

for img in bbox_pneumonia_id:
    os.rename('images/'+img, './data_img/pneumonia/'+img)

for img in bbox_normal_id:
    os.rename('images/'+img, './data_img/normal/'+img)
