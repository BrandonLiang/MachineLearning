#!/bin/python

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

# sigmoid function (for Neural Network feed-forward)
def sigmoid(x):
    return 1.0/(1+ np.exp(-x.astype(float)))

# binary classification - works closely with the output from sigmoid function
# domain specifies the two possible outcomes
def binary_classify(x, domain):
    condlist = [x < 0.5, x >= 0.5]
    choicelist = [domain[0], domain[1]]
    return np.select(condlist, choicelist)

# first derivative of sigmoid function (for Neural Network back-propagation)
def sigmoid_derivative(x):
    return x * (1.0 - x)

# convert csv file into a 2-dimensional numpy array
def read_csv_as_np(filepath, delim):
    return np.genfromtxt(filepath, delimiter = delim)

# Spark DataFrame -> Spark RDD -> Numpy Arrays & vice versa
# https://stackoverflow.com/questions/54190994/how-to-convert-spark-rdd-to-a-numpy-array
# https://stackoverflow.com/questions/36198264/convert-numpy-matrix-into-pyspark-rdd
class Spark_Function:
    def __init__(self, filepath):
      self.sparkSession = SparkSession.builder.getOrCreate()

      #self.sparkConf = sparkConf().setMaster("local[4]")
        # local[4]: 4 local cores
        # local:
        # spark://master:7077

      self.df = self.sparkSession.read.option("header", "true").option("inferSchema", "false").csv(filepath)
      self.rdd = self.df.rdd

    # convert csv file into a Spark RDD (Resilient Distributed Dataset)
    def to_rdd(self):
      return self.rdd

    # convert csv file into a Spark DataFrame
    def to_df(self):
      return self.df

    def df_show(self, n = 10, s = True):
      self.df.show(n, s)
      return

    # convert csv file into a Spark DataSet - need RowEncoder
    # https://jaceklaskowski.gitbooks.io/mastering-spark-sql/spark-sql-DataFrame.html

    # convert csv file from a Spark RDD to a 2-dimensional numpy array
    def rdd_to_np(self):
      return np.array(self.rdd.collect())

class Pandas_Function:
  def __init__(self, filepath, nrows, low_memory = True):
    self.df = pd.read_csv(filepath, encoding = "utf-8", delimiter = ",", low_memory = low_memory, nrows = int(nrows))

  def df_show(self):
    print(self.df.to_string())
    return
