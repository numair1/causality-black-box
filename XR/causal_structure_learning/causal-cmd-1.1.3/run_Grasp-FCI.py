import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--significance", type = float)
parser.add_argument("--input_dir")
parser.add_argument("--output_dir")
args = parser.parse_args()
# Remove folder if exists, then create
for subsample in os.listdir(args.input_dir):
#    subsample_index = subsample.split("_")[1].replace(".csv","")
     run_str = "./Grasp-FCI.sh "+args.input_dir+"/"+subsample+" "+str(args.significance)+" "+args.output_dir+"/learned_graphs"
     os.system(run_str)
