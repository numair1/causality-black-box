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
    attr_matrix = {'cardiomegaly': 0, 'atelectasis' : 1, 'effusion': 2,\
                   'infiltration': 3, 'mass': 4, 'nodule': 5, 'pneumothorax': 6,\
                    "y_pred": 7}
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
    tracking_dict = {'cardiomegaly': 0, 'atelectasis' : 0, 'effusion': 0,\
                     'infiltration': 0, 'mass': 0, 'nodule': 0, 'pneumothorax': 0,\
                    'y_pred': 0}
    for graph in os.listdir("./FCI_XR_01/learned_graphs"):
        if graph == ".DS_Store":
            continue
        edges = load_file("./FCI_XR_01/learned_graphs/"+graph)
        att_matrix = attr_matrix()
        count_rel_edges(edges, tracking_dict, att_matrix)
        #fname = get_file_name("./../FCI_May24_00001/learned_graph/"+graph)
        #write_to_file(adj_mat, fname)
    print(tracking_dict)
if __name__ == '__main__':
    main()
