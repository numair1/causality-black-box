import os
import numpy as np
import pprint
from statistics import mode
def create_outfile(features_dict):
    outfile = open("birds_dataset.txt", "w+")
    attr_list = []
    for im in features_dict:
        for feature_group in features_dict[im]:
            attr_list.append(feature_group.replace("has_",""))
        break
    attr_string = "\t".join(attr_list)
    outfile.write("image_name\t"+ attr_string +"\ty_pred\ty_true\n")
    return outfile

def get_true_y(image_name):
    if "Flycatcher" in image_name:
        true_label = 0
    elif "Gull" in image_name:
        true_label = 1
    elif "Kingfisher" in image_name:
        true_label = 2
    elif "Sparrow" in image_name:
        true_label = 3
    elif "Tern" in image_name:
        true_label = 4
    elif "Vireo" in image_name:
        true_label = 5
    elif "Warbler" in image_name:
        true_label = 6
    elif "Woodpecker" in image_name:
        true_label = 7
    elif "Wren" in image_name:
        true_label = 8
    return true_label

def collect_non_zero_values(value_list, ft_dict):
    non_zero_values = []
    i = 0
    for val in value_list:
        if val != 0:
            non_zero_values.append(ft_dict.keys()[i])
        i += 1
    return non_zero_values

def get_coarse_grouping(present_attr, feature_group):
    # Create coarse feature groupings
    coarse_mapping = {}
    coarse_mapping["has_bill_shape"] ={}
    coarse_mapping["has_bill_shape"]["curved_(up_or_down)"] = 1
    coarse_mapping["has_bill_shape"]["hooked"] = 1
    coarse_mapping["has_bill_shape"]["hooked_seabird"] = 1
    coarse_mapping["has_bill_shape"]["dagger"] = 2
    coarse_mapping["has_bill_shape"]["needle"] = 2
    coarse_mapping["has_bill_shape"]["cone"] = 2
    coarse_mapping["has_bill_shape"]["specialized"] = 3
    coarse_mapping["has_bill_shape"]["spatulate"] = 3
    coarse_mapping["has_bill_shape"]["all-purpose"] = 5

    coarse_mapping["color"] = {}
    coarse_mapping["color"]["blue"] = 1
    coarse_mapping["color"]["yellow"] = 1
    coarse_mapping["color"]["red"] = 1
    coarse_mapping["color"]["green"] = 2
    coarse_mapping["color"]["olive"] = 2
    coarse_mapping["color"]["purple"] = 2
    coarse_mapping["color"]["orange"] = 2
    coarse_mapping["color"]["pink"] = 2
    coarse_mapping["color"]["buff"] = 2
    coarse_mapping["color"]["iridescent"] = 2
    coarse_mapping["color"]["rufous"] = 3
    coarse_mapping["color"]["grey"] = 3
    coarse_mapping["color"]["black"] = 3
    coarse_mapping["color"]["brown"] = 3
    coarse_mapping["color"]["white"] = 4

    coarse_mapping["pattern"] = {}
    coarse_mapping["pattern"]["solid"] = 1
    coarse_mapping["pattern"]["spotted"] = 2
    coarse_mapping["pattern"]["striped"] = 3
    coarse_mapping["pattern"]["multi-colored"] = 4

    coarse_mapping["has_tail_shape"] = {}
    coarse_mapping["has_tail_shape"]["forked_tail"] = 1
    coarse_mapping["has_tail_shape"]["rounded_tail"] = 2
    coarse_mapping["has_tail_shape"]["notched_tail"] = 3
    coarse_mapping["has_tail_shape"]["fan-shaped_tail"] = 4
    coarse_mapping["has_tail_shape"]["pointed_tail"] = 5
    coarse_mapping["has_tail_shape"]["squared_tail"] = 6

    coarse_mapping["has_bill_length"] = {}
    coarse_mapping["has_bill_length"]["about_the_same_as_head"] = 1
    coarse_mapping["has_bill_length"]["longer_than_head"] = 2
    coarse_mapping["has_bill_length"]["shorter_than_head"] = 3

    coarse_mapping["has_wing_shape"] = {}
    coarse_mapping["has_wing_shape"]["rounded-wings"] = 1
    coarse_mapping["has_wing_shape"]["pointed-wings"] = 2
    coarse_mapping["has_wing_shape"]["broad-wings"] = 3
    coarse_mapping["has_wing_shape"]["tapered-wings"] = 4
    coarse_mapping["has_wing_shape"]["long-wings"] = 5

    coarse_mapping["has_size"] = {}
    coarse_mapping["has_size"]["large_(16_-_32_in)"] = 1
    coarse_mapping["has_size"]["very_large_(32_-_72_in)"] = 1
    coarse_mapping["has_size"]["small_(5_-_9_in)"] = 2
    coarse_mapping["has_size"]["very_small_(3_-_5_in)"] = 2
    coarse_mapping["has_size"]["medium_(9_-_16_in)"] = 3

    if "color" in feature_group:
        coarse_key = "color"
    elif "pattern" in feature_group:
        coarse_key = "pattern"
    else:
        coarse_key = feature_group
    # if len(present_attr) == 1, life is easy
    if len(present_attr) == 1:
        return coarse_mapping[coarse_key][present_attr[0]]
    else:
        coarse_hits = []
        for attr in present_attr:
            coarse_hits.append(coarse_mapping[coarse_key][attr])
            return mode(coarse_hits)

# Create dictionary mapping image_id to image_name
im_id_dict = {}
with open("./../data/CUB_200_2011/CUB_200_2011/images.txt", "r") as infile:
    for line in infile.readlines():
        im_id, im_name = line.split()
        im_name = im_name.split("/")[-1]
        im_id_dict[str(im_id)] = im_name

# Create dictionary mapping attribute_id to attribute_name
attribute_id_dict = {}
with open("./../data/CUB_200_2011/attributes.txt") as infile:
    for line in infile.readlines():
        attr_id, attr_name = line.split()
        attr_name = attr_name
        attribute_id_dict[str(attr_id)] = attr_name

# Create dictionary mapping image_name to predicted value
im_pred_dict = {}
with open("./../network_training/model_preds.txt") as infile:
    for line in infile.readlines():
        im, pred = line.split(",")
        im_pred_dict[im] = pred

# Let's do the magic
features_dict = {}
with open("./../data/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt", "r") as infile:
    for line in infile.readlines():
        im_id, attr_id, present, certainty, time = line.split()
        img_name = im_id_dict[im_id]
        attr_group_name, attr_fine_name = attribute_id_dict[attr_id].split("::")
        attr_present =  present
        if attr_group_name == "has_head_pattern" or attr_group_name == "has_shape":
            continue
        else:
            if img_name not in features_dict:
                features_dict[img_name] = {}
            if attr_group_name not in features_dict[img_name]:
                features_dict[img_name][attr_group_name] = {}
            features_dict[img_name][attr_group_name][attr_fine_name] = int(attr_present)

outfile = create_outfile(features_dict)
for im in features_dict:
    if im in im_pred_dict:
        line_str = im
        y_pred = im_pred_dict[im]
        y_true = get_true_y(im)
        for feature_group in features_dict[im]:
            ft_dict = features_dict[im][feature_group]
            label = 0
            # Build in resolving of draws
            if 1 in ft_dict.values():
                value_list = list(ft_dict.values())
                present_attr = collect_non_zero_values(value_list, ft_dict)
                label = get_coarse_grouping(present_attr, feature_group)
            line_str += "\t"+str(label)
        line_str += "\t" + str(y_pred.replace("\n", "")) + "\t" + str(y_true)
        outfile.write(line_str + "\n")
