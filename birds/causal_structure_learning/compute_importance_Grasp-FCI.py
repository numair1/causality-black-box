import numpy as np
import os

def get_file_name(file):
    with open(file, "r") as infile:
        for line in infile.readlines():
            if "file:" in line:
                file_name = line.split(" ")[1]
                file_name = file_name.replace(".csv","")
                file_name = file_name.replace("\n","")
                return file_name

# Parse in file in raw text
def load_file(file):
    relevant_lines = []
    with open(file, "r") as infile:
        start = False
        for line in infile.readlines():
            line = line.replace("\n", "")
            if start:
                relevant_lines.append(line)
            if line == "Graph Edges:":
                start = True
    return relevant_lines

# Create matrix mapping attribute_name to index
def attr_matrix():
    attr_matrix = {"wing_shape": 0, "back_pattern": 1, "upper_tail_color": 2, \
                    "bill_shape": 3, "upperparts_color": 4, "underparts_color": 5, \
                    "under_tail_color": 6, "tail_shape": 7, "throat_color": 8, \
                    "wing_pattern": 9, "nape_color": 10, "belly_color": 11, \
                    "tail_pattern": 12, "belly_pattern": 13, "primary_color": 14,\
                    "leg_color": 15, "bill_length": 16, "bill_color": 17,\
                    "size": 18, "crown_color": 19, "wing_color": 20,\
                    "back_color": 21, "breast_color":22,"breast_pattern":23,\
                    "eye_color": 24, "forehead_color": 25, "y_pred": 26}
    return attr_matrix

# Create 2-D adjacency matrix that is output
def count_rel_edges(relevant_lines, tracking_matrix, att_matrix):
    adj_mat = np.zeros((27,27))
    for line in relevant_lines:
        start_vert, edge, end_vert = line.split(" ")[1:4]
        if end_vert == "y_pred" and edge == "o->":
            tracking_matrix[start_vert] += 1
        elif end_vert == "y_pred" and edge == "-->":
            tracking_matrix[start_vert] += 1
def main():
    tracking_dict = {"wing_shape": 0, "back_pattern": 0, "upper_tail_color": 0, \
                    "bill_shape": 0, "upperparts_color": 0, "underparts_color": 0, \
                    "under_tail_color": 0, "tail_shape": 0, "throat_color": 0, \
                    "wing_pattern": 0, "nape_color": 0, "belly_color": 0, \
                    "tail_pattern": 0, "belly_pattern": 0, "primary_color": 0,\
                    "leg_color": 0, "bill_length": 0, "bill_color": 0,\
                    "size": 0, "crown_color": 0, "wing_color": 0,\
                    "back_color": 0, "breast_color": 0,"breast_pattern": 0,\
                    "eye_color": 0, "forehead_color": 0, "y_pred": 0}
    for graph in os.listdir("./Grasp-FCI/learned_graphs"):
        if graph == ".DS_Store":
            continue
        edges = load_file("./Grasp-FCI/learned_graphs/"+graph)
        att_matrix = attr_matrix()
        count_rel_edges(edges, tracking_dict, att_matrix)
    print(tracking_dict)
if __name__ == '__main__':
    main()
