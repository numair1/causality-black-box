import os
import shutil
import numpy as np

# loop through outer Directory
for bird in os.listdir("./../data/consolidated_dataset/"):
    if bird == ".DS_Store":
        continue
    print(bird + ":"+ str(len(os.listdir("./../data/consolidated_dataset/"+bird))))

# Sparrow and Warbler are overrepresnted and need to be dropped
for image in os.listdir("./../data/consolidated_dataset/Sparrow"):
    if np.random.uniform(low = 0, high = 1, size = 1)>0.31:
        os.remove("./../data/consolidated_dataset/Sparrow/"+image)

# Sparrow and Warbler are overrepresnted and need to be dropped
for image in os.listdir("./../data/consolidated_dataset/Warbler"):
    if np.random.uniform(low = 0, high = 1, size = 1)>0.25:
        os.remove("./../data/consolidated_dataset/Warbler/"+image)

for bird in os.listdir("./../data/consolidated_dataset/"):
    if bird == ".DS_Store":
        continue
    print(bird + ":"+ str(len(os.listdir("./../data/consolidated_dataset/"+bird))))
