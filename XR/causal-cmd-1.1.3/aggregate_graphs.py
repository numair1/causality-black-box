import numpy as np
import os
import json

def parse_graph_txt(graph_edges, subsample_index):
    agg_dict = {}
    for key in graph_edges:
        unique_items, counts = np.unique(graph_edges[key],return_counts = True)
        unique_items = list(unique_items)
        counts = list(counts)
        rel_votes = 0
        if "net_pred" in key:
            if "o->" in unique_items:
                ambig_arr_index = unique_items.index("o->")
                rel_votes+= counts[ambig_arr_index]
            if "-->" in unique_items:
                direct_index = unique_items.index("-->")
                rel_votes+= counts[direct_index]
            if rel_votes>= 3:
                agg_dict[key] = "-->"
        else:
            if "o->" in unique_items:
                ambig_arr_index = unique_items.index("o->")
                rel_votes+= counts[ambig_arr_index]
            if "<->" in unique_items:
                bidirect_index = unique_items.index("<->")
                rel_votes+= counts[bidirect_index]
            if rel_votes>= 3:
                agg_dict[key] = "<->"
    # Save to file
    json_str = json.dumps(agg_dict)
    with open("./../data/agg_graph_txt/graph_"+str(subsample_index)+".txt", "w+") as outfile:
        outfile.write(json_str)
    return agg_dict

def parse_graph_mat(graph_edges, subsample_index):
    adj_mat = np.zeros((44,44))
    edge_index_map = {"Black_Hair":0,"Blond_Hair":1,"Brown_Hair":2,"Bald":3,\
                    "Mustache":4,"Smiling":5,"Frowning":6,"Chubby":7,\
                    "Curly_Hair":8,"Wavy_Hair":9, "Straight_Hair":10,\
                    "Receding_Hairline":11,"Bangs":12,"Sideburns":13,\
                    "Bushy_Eyebrows":14,"Arched_Eyebrows":15,"Narrow_Eyes":16,\
                    "Big_Nose":17, "Pointy_Nose":18,"Big_Lips":19,"Mouth_Closed":20,\
                    "No_Beard":21,"Goatee":22,"Round_Jaw":23,"Double_Chin":24,\
                    "Oval_Face":25,"Square_Face":26,"Round_Face":27,"Gray_Hair":28,\
                    "Bags_Under_Eyes":29,"Heavy_Makeup":30,"Rosy_Cheeks":31,\
                    "Shiny_Skin":32, "Pale_Skin":33,"5_o_Clock_Shadow":34,\
                    "Strong_Nose-Mouth_Lines":35,"Wearing_Lipstick":36,"Flushed_Face":37,\
                    "High_Cheekbones":38,"Brown_Eyes":39,"Wearing_Earrings":40,\
                    "Wearing_Necktie":41, "Wearing_Necklace":42,"net_pred":43\
                    }
    for edge in graph_edges:
        start_edge, end_edge = edge.split(",")
        start_ind = edge_index_map[start_edge]
        end_ind = edge_index_map[end_edge]
        arrow = graph_edges[edge]
        if arrow == "<->":
            adj_mat[start_ind,end_ind] = 2
            adj_mat[end_ind,start_ind] = 2
        elif arrow == "-->":
            adj_mat[start_ind,end_ind] = 2
            adj_mat[end_ind,start_ind] = 3
    with open("./../data/agg_graph_adj_mat/agg_graph_mat_"+str(subsample_index)+".txt", "w+") as outfile:
        for i in range(len(adj_mat)):
            line_str = ""
            j = 0
            for val in adj_mat[i,:]:
                if j == len(adj_mat[i,:]) -1:
                    line_str+= str(val)
                else:
                    line_str+= str(val)+"\t"
                    j+=1
            outfile.write(line_str+"\n")

def check_valid_line(line):
    if ("<->" in line) or ("o->" in line) or ("-->" in line) or ("o-o" in line):
        return True
    return False
def extract_data(line):
     start_vert, edge, end_vert = line.split(" ")
     return start_vert, edge, end_vert

def insert_into_dict(start_edge, arrow, end_edge, adj_dict):
    vertex_str = start_edge+","+end_edge
    if vertex_str not in adj_dict:
        adj_dict[vertex_str] = []
    adj_dict[vertex_str].append(arrow)

def get_subsample_graphs(subsample_index):
    file_names = []
    for graph in os.listdir("./../data/indiv_graphs_txt/"):
        if "_"+str(subsample_index)+"_" in graph:
            file_names.append(graph)
    return file_names

def read_process_graphs(file_names):
    graph_edges = {}
    for graph_file in file_names:
        with open("./../data/indiv_graphs_txt/"+graph_file,"r") as infile:
            for line in infile.readlines():
                line = line.replace("\n","")
                if len(line) == 0:
                    continue
                if check_valid_line(line):
                    insert_into_dict(line.split(" ")[1], line.split(" ")[2], line.split(" ")[3],graph_edges)
    return graph_edges

def main():
    for i in range(0,50):
        file_names = get_subsample_graphs(i)
        graph_edges = read_process_graphs(file_names)
        agg_dict = parse_graph_txt(graph_edges,i)
        parse_graph_mat(agg_dict,i)

if __name__ == '__main__':
    main()
