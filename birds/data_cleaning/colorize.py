import os
from skimage import color
from skimage import io

# Delete old garbage tries
if "train" in os.listdir("./../real_data/color_data"):
    os.rmdir("./../real_data/color_data/train")

if "test" in os.listdir("./../real_data/color_data"):
    os.rmdir("./../real_data/color_data/test")

if "val" in os.listdir("./../real_data/color_data"):
    os.rmdir("./../real_data/color_data/val")

for split in os.listdir("./../real_data/final_data/"):
    if split == ".DS_Store":
        os.remove("./../real_data/final_data/"+".DS_Store")
    for species in os.listdir("./../real_data/final_data/"+split):
        if species == ".DS_Store":
            os.remove("./../real_data/final_data/"+split+"/"+".DS_Store")
        for image in os.listdir("./../real_data/final_data/"+split+"/"+species):
            if image == ".DS_Store":
                os.remove("./../real_data/final_data/"+split+"/"+species+"/"+".DS_Store")
            im = io.imread("./../real_data/final_data/"+split+"/"+species+"/"+image)
            if len(im.shape)!= 3:
                os.remove("./../real_data/final_data/"+split+"/"+species+"/"+image)
                print(im.shape)
            elif im.shape[2] != 3:
                os.remove("./../real_data/final_data/"+split+"/"+species+"/"+image)
                print(im.shape)
