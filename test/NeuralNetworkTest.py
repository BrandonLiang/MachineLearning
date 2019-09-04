#!/bin/python
import sys
import numpy as np

sys.path.insert(0, "../src/")
from NeuralNetwork import NeuralNetwork # since NeuralNetwork class is inside the module/file NeuralNetwork.py
from functions import *

# dimension of weights1 and weights2?
# number of layers? (will be more weights)
# iterations?

# read in file and output a file with prediction

# cv: training, validation & testing set
# accuracy & precision: confusion matrix (for binary classificaiton)
# accuracy

# feature significance, entropy?

# /Users/brandonliang/external/NBA_Data/NN_Each_Game_Distribution_4_15_ready.csv
# /Users/brandonliang/external/NBA_Data/NN_Each_Game_Distribution_4_15.csv

#data = read_csv_as_np("/Users/brandonliang/external/NBA_Data/NN_Each_Game_Distribution_4_15.csv", ',')[1:,:] # remove header
#print(data[:,:-1])
#spark_function = Spark_Function("/Users/brandonliang/external/NBA_Data/NN_Each_Game_Distribution_4_15.csv")
#spark_function.to_df().show()

'''
filepath = "/Users/brandonliang/src/MachineLearning/csv/first_column.csv"
header = 'infer'
label_transformation = None
'''

filepath = "/Volumes/external/NBA_Data/NN_Each_Game_Distribution_4_15_ready.csv"
header = None
label_transformation = "L"
'''
x = np.array([[0,0,0],
              [0,0,1],
              [0,1,0],
              [0,1,1],
              [1,0,0],
              [1,0,1],
              [1,1,1],
              [1,1,0]])
y = np.array([[0], [0], [0], [0], [1], [1], [1], [1]])
'''

#dimension = int(sys.argv[1])
dimension = [4, 4]
iteration = int(sys.argv[1])
#nn = NeuralNetwork(x, y, dimension, iteration)
nn = NeuralNetwork(filepath, -1, dimension, iteration, infer_header = header, label_transformation = label_transformation)

nn.train()
#print(nn.summary())
#print(nn.total_error())
#print(nn.sme())
