import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", type = int)
parser.add_argument("-f")
parser.add_argument("-o")
args = parser.parse_args()

data = pd.read_csv(args.f, sep = "\t")
#data = data.drop(['image_name', 'y_true'], axis = 1)
n_subsamples = args.n
for i in range(n_subsamples):
    i_subsample = data.sample(frac = 1, replace = True)
    i_subsample.to_csv(args.o + "/subsample_"+str(i)+".csv",sep = '\t', index = False)
