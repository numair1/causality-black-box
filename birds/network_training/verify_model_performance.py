import os

def create_test_list():
    im_list = []
    for label in os.listdir("./../real_data/final_data/train"):
        if label == ".DS_Store":
            continue
        for im in os.listdir("./../real_data/final_data/train/"+ label):
            if im == ".DS_Store":
                continue
            im_list.append(im)
    return im_list

im_list = create_test_list()
with open("./model_preds.txt", "r") as infile:
    correct = 0
    total = 0
    for image in infile.readlines():
        img_name, pred = image.split(",")
        if img_name not in im_list:
	        continue

        pred = int(pred)
        if "Flycatcher"  in img_name:
            true_label = 0
        elif "Gull" in img_name:
            true_label = 1
        elif "Kingfisher" in img_name:
            true_label = 2
        elif "Sparrow" in img_name:
            true_label = 3
        elif "Tern" in img_name:
            true_label = 4
        elif "Vireo" in img_name:
            true_label = 5
        elif "Warbler" in img_name:
            true_label = 6
        elif "Woodpecker" in img_name:
            true_label = 7
        elif "Wren" in img_name:
            true_label = 8
        if pred == true_label:
            correct += 1
        total += 1.0
    print(correct/total)
    print(len(im_list))
