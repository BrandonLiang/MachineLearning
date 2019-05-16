#!/bin/python
import numpy as np
import functions
import sys

# dimension of weights1 and weights2?
# number of layers? (will be more weights)
# iterations?

# read in file and output a file with prediction
# Spark DataFrame -> Spark RDD -> Numpy Arrays & vice versa
# https://stackoverflow.com/questions/54190994/how-to-convert-spark-rdd-to-a-numpy-array
# https://stackoverflow.com/questions/36198264/convert-numpy-matrix-into-pyspark-rdd

# cv: training, validation & testing set
# accuracy & precision: confusion matrix (for binary classificaiton)
# accuracy

# feature significance, entropy?

class NeuralNetwork:
  def __init__(self, feature, label, dimension):
    self.input = feature
    self.weights1 = np.random.rand(self.input.shape[1], dimension)
    self.weights2 = np.random.rand(dimension, 1)
    self.label = label
    self.output = np.zeros(self.label.shape)

  def feedforward(self):
    self.layer1 = functions.sigmoid(np.dot(self.input, self.weights1))
    self.output = functions.sigmoid(np.dot(self.layer1, self.weights2))

  def backprop(self):
    # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    d_weights2 = np.dot(self.layer1.T, (2*(self.label - self.output) * functions.sigmoid_derivative(self.output)))
    d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.label - self.output) * functions.sigmoid_derivative(self.output), self.weights2.T) * functions.sigmoid_derivative(self.layer1)))

    # update the weights with the derivative (slope) of the loss function
    self.weights1 += d_weights1
    self.weights2 += d_weights2

if __name__ == "__main__":
  x = np.array([[0,0,0],
                [0,0,1],
                [0,1,0],
                [0,1,1],
                [1,0,0],
                [1,0,1],
                [1,1,1],
                [1,1,0]])
  y = np.array([[0], [0], [0], [0], [1], [1], [1], [1]])

  dimension = int(sys.argv[1])
  iteration = int(sys.argv[2])
  nn = NeuralNetwork(x, y, dimension)

  for i in range(iteration):
    nn.feedforward()
    nn.backprop()

  nn.feedforward()
  print(nn.output)
