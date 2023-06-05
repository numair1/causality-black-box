import os
import shutil
# loop through outer Directory
for bird in os.listdir("./../data/consolidated_dataset/"):
    if bird == ".DS_Store":
        continue
    for sub_bird in os.listdir("./../data/consolidated_dataset/"+bird):
        if sub_bird == ".DS_Store":
            continue
        for image in os.listdir("./../data/consolidated_dataset/"+bird+"/"+sub_bird):
            if image == ".DS_Store":
                continue
            shutil.move("./../data/consolidated_dataset/"+bird+"/"+sub_bird+"/"+image,\
                        "./../data/consolidated_dataset/"+bird+"/"+image)
