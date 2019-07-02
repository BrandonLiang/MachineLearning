#!/bin/python
import numpy as np
import functions

# dimension of weights1 and weights2?
# number of layers? (will be more weights)
# iterations?

# hidden layer
# bias (disconnected layers)

# read in file and output a file with prediction

# methods:
# cv: training, validation & testing set
# split to train, cv & testing set
# accuracy & precision: confusion matrix (for binary classificaiton)
# accuracy

# feature significance, entropy?

class NeuralNetwork:
  def __init__(self, feature, label, dimension, iterations):
    self.input = feature
    self.weights1 = np.random.rand(self.input.shape[1], dimension)
    self.weights2 = np.random.rand(dimension, 1)
    self.label = label
    self.iterations = iterations
    self.output = np.zeros(self.label.shape)
  
  # applying existing Neural Network weights (model hyper-parameters) on input data to validate against real labels
  def feedforward(self):
    self.layer1 = functions.sigmoid(np.dot(self.input, self.weights1))
    self.output = functions.sigmoid(np.dot(self.layer1, self.weights2))

  # training the Neural Network model by updating the weights (model hyper-parameters) from the real labels against previously applied input data
  def backprop(self):
    # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    d_weights2 = np.dot(self.layer1.T, (2*(self.label - self.output) * functions.sigmoid_derivative(self.output)))
    d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.label - self.output) * functions.sigmoid_derivative(self.output), self.weights2.T) * functions.sigmoid_derivative(self.layer1)))

    # update the weights with the derivative (slope) of the loss function
    self.weights1 += d_weights1
    self.weights2 += d_weights2

  def train(self):
    for i in range(self.iterations):
      self.feedforward()
      self.backprop()

    self.feedforward()

  #def test(self):

  def summary(self):
    return self.output
