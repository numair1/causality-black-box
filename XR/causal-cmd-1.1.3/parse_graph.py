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
def create_adj_matrix(relevant_lines, attr_matrix):
    adj_mat = np.zeros((27,27))
    for line in relevant_lines:
        start_vert, edge, end_vert = line.split(" ")[1:4]
        #print(start_vert+" "+ edge + " " + end_vert)
        insert_into_mat(start_vert, edge, end_vert, adj_mat, attr_matrix)
    return adj_mat

def insert_into_mat(start_edge, arrow, end_edge, adj_mat, attr_matrix):
    start_ind = attr_matrix[start_edge]
    end_ind = attr_matrix[end_edge]
    if arrow == "<->":
        adj_mat[start_ind,end_ind] = 2
        adj_mat[end_ind,start_ind] = 2
    elif arrow == "o->":
        if start_edge == "y_pred":
            adj_mat[start_ind,end_ind] = 2
            adj_mat[end_ind,start_ind] = 2
        else:
            adj_mat[start_ind,end_ind] = 2
            adj_mat[end_ind,start_ind] = 1
    elif arrow == "-->":
        adj_mat[start_ind,end_ind] = 2
        adj_mat[end_ind,start_ind] = 3
    elif arrow == "o-o":
        if start_edge == "y_pred":
            adj_mat[start_ind,end_ind] = 1
            adj_mat[end_ind,start_ind] = 2
        elif end_edge == "y_pred":
            adj_mat[start_ind,end_ind] = 2
            adj_mat[end_ind,start_ind] = 1
        else:
            adj_mat[start_ind,end_ind] = 1
            adj_mat[end_ind,start_ind] = 1

def write_to_file(adj_mat, file_name):
    with open("./../FCI_May24_00001/adjacency_matrix/"+ file_name + ".txt", "w+") as outfile:
        for row in adj_mat:
            row = row.astype(str)
            outfile.write(" ".join(row)+"\n")
# adj_mat = np.zeros((10,10))
# for child in root:
#     for vars in child:
#         if child.tag == "variables":
#              continue
#         else:
#             start_vert, edge, end_vert = extract_data(vars.text)
#             insert_into_mat(start_vert, edge, end_vert, adj_mat)
# with open("adj_mat.txt", "w+") as outfile:
#     for i in range(len(adj_mat)):
#         line_str = ""
#         for val in adj_mat[i,:]:
#             line_str+= str(val)+"\t"
#         outfile.write(line_str+"\n")

def main():
    for graph in os.listdir("./../FCI_May24_00001/learned_graph"):
        if graph == ".DS_Store":
            continue
        edges = load_file("./../FCI_May24_00001/learned_graph/"+graph)
        att_matrix = attr_matrix()
        adj_mat = create_adj_matrix(edges, att_matrix)
        fname = get_file_name("./../FCI_May24_00001/learned_graph/"+graph)
        write_to_file(adj_mat, fname)
if __name__ == '__main__':
    main()
