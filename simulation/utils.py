import pandas as pd
from csv import DictReader


def xor(input_a, input_b):
    if input_a == '0' and input_b == '0':
        return 0
    elif input_a == '0' and input_b == '1':
        return 1
    elif input_a == '1' and input_b == '0':
        return 1
    elif input_a == '1' and input_b == '1':
        return 1


def xor_model_predictions(h_bar_predictions, v_bar_predictions):
    combined_predictions = {}
    for key in h_bar_predictions:
        combined_predictions[key] = {"xor_prediction": xor(h_bar_predictions[key]["predicted_label"],
                                                           v_bar_predictions[key]["predicted_label"]),
                                     "h_bar_network_prediction": h_bar_predictions[key]["predicted_label"],
                                     "v_bar_network_prediction": v_bar_predictions[key]["predicted_label"]
                                     }
    return combined_predictions


def merge_xor_predictions_with_hand_crafted_features(combined_predictions, hand_crafted_ft_file):
    """
    Function that takes in file name for handcrafted features and dictionary of image predictions and returns pandas
    dataframe containing handcrafted features and predictions
    :param combined_predictions:
    :param hand_crafted_ft_file:
    :return:
    """
    df = pd.DataFrame(columns=["img_name", "h_bar", "v_bar", "circle", "triangle", "y", "xor_prediction",
                               "h_bar_network_prediction", "v_bar_network_prediction"])
    with open(hand_crafted_ft_file, "r") as infile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_dict = {"img_name": row["img_name"], "h_bar": row["h_bar"], "v_bar": row["v_bar"],
                        "circle": row["circle"], "triangle": row["triangle"], "y": row["y"],
                        "xor_prediction": combined_predictions["xor_prediction"],
                        "h_bar_network_prediction": combined_predictions["h_bar_network_prediction"],
                        "v_bar_network_prediction": combined_predictions["v_bar_network_prediction"]}
            df.append(row_dict)
    return df