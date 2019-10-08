#!/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import functions

# to understand how a Neural Network works
# to use my understanding of Neural Network parameters to optimize customized Neural Network models given any subject matter

# save and load weights + bias + learning rate as model hyperparameters

# use Matplotlib to graph E_total vs iterations

# dimension of weights1 and weights2?
# number of layers? (will be more weights)
# iterations?

# hidden layer
# bias (disconnected layers) - how to calculate the bias weight?
# learning rate (alpha) - how to calculate the learning rate?
# training result threshold? - when E_total gets below certain level, training stops?
# - make it interactive!

# experiment with different sets of parameters (number of layers, number of hidden neurons, learning rate, bias)

# read in file and output a file with prediction

# *training, k-fold cv, testing set!!!

# methods:
# *cv: training, validation & testing set -- read more about purpose and process of CV!
# split to train, cv & testing set
# accuracy & precision: confusion matrix (for binary classificaiton)
# accuracy

# feature significance, entropy? -- yes, add this!

# save plot fig

# generalize hidden layers and weights based on $dimension
# bias at each weight
# Binary Classify Error Compute! + Confusion Matrix, Precision, Accuracy (done)
# Reverse Label Transformation for Both Label & Output
# Predict with original label range (concat onto testing data, on top of actual label column -- ......,label,output)

# *complete NN backpropagation matrix operation write up in notebook! -- see Matrix Calculus in bookmarked web under "Course" in firefox

# Save NN Model
# Run on Web via Click - fullstack -- Web needs to have Java running!
# NN for multi-class classification? i.e.: image recognition? object recognition? hand-written character recognition?

# precision + recall - confusion matrix
# refer to 4705/hw0 client code for precision and recall functions + plots

# see decision tree reading of 4771/Lecture 1, Entropy!
# kd-tree, random forest, deep forest

'''
more ides (improvement, features, future research):
  1. sklearn (and also check pytorch) sklearn_metric for confusion matrix, accuracy, precision, recall, f1_score, etc.
  2. CV
  3. Training (CV), Dev (Hyperparameter Tuning), Test
  4. Word Embedding
  5. Bag of words
  6. NN for NLP (Einstein + NN4NLP books)
  7. Deep Learning, Reinforcement Learning, Recurrent Neural Network, Convolutional Neural Network
  8. ReLu, tanh, softmax, sigmoid - activation function (non-linear)
  9. Gradient Descent Formula (for Generalizing Backprop with arbitray numbers of hidden layers)
  10. Stochastic Gradient Descent, Minibatch Stochastic Gradient Descent
  11. * Learning rate (initial 10e-3), decrease learning rate with each epoch
  12. Epoch, batch, iteration
  13. Add attribute .index to retrieve location (other attributes as well)
  14. * predict, see sklearn_metrics
  15. Write out Forward and BackPropagation Matrix Operation, Gradient Descent, linear combination, etc.
  16. * Initialize weight
    - Xavier initialization
    - N(0,1)
    - Uniform(-1,1)
    The last two are kind of bad random draws because they may result in poetential dead neurons (?)
  16.1 * Initialize Weight Research: https://hackernoon.com/how-to-initialize-weights-in-a-neural-net-so-it-performs-well-3e9302d4490f
    - Relu & Tanh -> using N(0,1) & multiply the sample with square root of (1/ni) where ni is the number of input units for that layer
    - Tanh in output no term -> using N(0,1) & multiply the sample with square root of (1/(ni + no)) where (ni + no) is the number of input units for that layer
      W = np.random.rand((x_dim,y_dim))*np.sqrt(1/(ni+no))
  17. * Regularization (L1, L2, which is Lasso & Ridge and their characteristics, adv & disadv)
  18. * partial_fit (see sklearn)
  19. Optimizing Gradient Descent
    ruder.io/optimizing-gradient-descent/
    - Adagrad
    - Adaboost
    - etc.
  20. Drop Out - YMMV for NLP
  21. Hinge Loss
  22. Multi-class / binary entropy loss
  23. entropy
  24. ReLu, Tanh, softmax, sigmoid

Research:
  1. importantce of hidden layer
  2. systematic and theoretical way of deciding # of hidden layers and # of neurons/nodes for each layer (except for input & output layers)
'''

class NeuralNetwork:
  def __init__(self, file_input, label_index, dimension, iterations, infer_header = 'infer', label_transformation = None):
    '''
    input
      label_index: the index of the label column in the dataset; for example, if it's the last column, label_index = -1
      dimension: list of int, with each element equaling the number of nodes for its corresponding hidden layer; length of the list equals number of hidden layers
      label_transformation: the string equivalent of "0.0" of the labels; default is None, when no label transformation is necessary
    '''
    if (type(file_input) is pd.core.frame.DataFrame):
      self.df = dataframe
    elif (type(file_input) is str):
      self.df = pd.read_csv(filepath, header = infer_header)
    else:
      print("must have one input from dataframe or filepath") # sys error - stderr
    if label_transformation is not None: # need to perform label transformation from string to float
      # need to reorder to ensure label_transformation is always at index 0 in this list!
      self.original_label_range = list(np.unique(self.df.values[:, label_index]))
      self.original_label_range.remove(label_transformation)
      self.original_label_range.insert(0, label_transformation)

      self.column_size = len(self.df.columns)
      self.df.iloc[:,label_index] = self.df.apply(lambda x: 0.0 if x[self.column_size + label_index] == label_transformation else 1.0, axis = 1)
    self.ndarray = self.df.values # convert from pandas df to numpy ndarray
    self.input = self.ndarray[:, : label_index]
    self.size = self.input.shape[0] # number of rows of training data
    self.label = np.reshape(self.ndarray[:, label_index], (self.size, 1)) # has to reshape label into a 2d array
    self.label_range = list(np.unique(self.label)) # always sorted
    # dimension means the number of hidden neurons in each hidden layer, must be a list; len(list) = number of layers
    self.hidden_layers = len(dimension) # number of hidden layers based on dimension input
    # always one more weight than number of hidden layers
    self.weights1 = np.random.rand(self.input.shape[1], dimension[0])
    self.weights2 = np.random.rand(dimension[0], dimension[1])
    self.weights3 = np.random.rand(dimension[1], self.label.shape[1])
    #self.bias = np.random.rand() # bias, y-intercept
    self.iterations = iterations
    self.output = np.zeros(self.label.shape)
    self.output_classified = None
    self.plot = False # no graph by default
    self.iteration = 0
    self.start = None
    self.end = None
    self.time = None
  
  '''
  # applying existing Neural Network weights (model hyper-parameters) on input data to validate against real labels
  def feedforward(self):
    self.layer1 = functions.sigmoid(np.dot(self.input, self.weights1))
    self.output = functions.sigmoid(np.dot(self.layer1, self.weights2))

  one weight: d_weights = np.dot(self.input.T,
                                 2 * (self.label - self.output) * functions.sigmoid_derivative(self.output)
                          )

  # training the Neural Network model by updating the weights (model hyper-parameters) from the real labels against previously applied input data
  def backprop(self):
    # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    d_weights2 = np.dot(self.layer1.T,
                        2 * (self.label - self.output) * functions.sigmoid_derivative(self.output)
                 )
    d_weights1 = np.dot(self.input.T,
                        np.dot(2 * (self.label - self.output) * functions.sigmoid_derivative(self.output),
                               self.weights2.T
                        ) * functions.sigmoid_derivative(self.layer1)
                 )

    # update the weights with the derivative (slope) of the loss function
    self.weights1 -= d_weights1
    self.weights2 -= d_weights2
  '''

  # with 2 hidden layers, start with 2 layers and 3 weights
  def feedforward(self):
    self.layer1 = functions.sigmoid(np.dot(self.input, self.weights1))
    self.layer2 = functions.sigmoid(np.dot(self.layer1, self.weights2))
    self.output = functions.sigmoid(np.dot(self.layer2, self.weights3))
    self.output_classified = functions.binary_classify(self.output, self.label_range) # make it flexible here!

  def backprop(self):
    d_weights3 = np.dot(self.layer2.T,
                        2 * (self.label - self.output) * functions.sigmoid_derivative(self.output)
                 )
    d_weights2 = np.dot(self.layer1.T,
                        np.dot(2 * (self.label - self.output) * functions.sigmoid_derivative(self.output),
                               self.weights3.T
                        ) * functions.sigmoid_derivative(self.layer2)
                 )
    d_weights1 = np.dot(self.input.T,
                        np.dot(np.dot(2 * (self.label - self.output) * functions.sigmoid_derivative(self.output),
                                      self.weights3.T
                               ) * functions.sigmoid_derivative(self.layer2),
                               self.weights2.T
                        ) * functions.sigmoid_derivative(self.layer1)
                 )

    # update the weights with the derivative (slope) of the loss function
    self.weights1 -= d_weights1
    self.weights2 -= d_weights2
    self.weights3 -= d_weights3

  def summary(self):
    return self.output

  def compute_total_error(self):
    self.total_error = self.label - self.output
    return self.total_error

  def compute_sme(self):
    self.sme = np.dot(self.total_error.T, self.total_error).item() * 1.0 / self.size # convert singleton array into a scalar for return
    print("Iteration {}: {}".format(self.iteration, self.sme))
    return self.sme

  def binary_classify(self):
    self.output_classified = functions.binary_classify(self.output, self.label_range)

  # confusion matrix + accuracy + precision
  def confusion_matrix(self):
    unique, self.output_classified_unique_count = np.unique(np.equal(self.output_classified, self.label), return_counts = True)
    self.output_classified_success = dict(zip(unique, self.output_classified_unique_count)).get(True)
    print("Correct predictions: {}".format(self.output_classified_success))
    print("Out of total inputs: {}".format(self.size))
    # self.output_classified vs self.label

  def reset_iteration(self):
    self.iteration = 0

  def time_start(self):
    self.start = time.time()

  def time_end(self):
    self.end = time.time()
    self.time = self.end - self.start

  def graph_sme(self, iteration):
      self.compute_total_error()
      self.compute_sme()
      if iteration == 0: # first iteration, set the plot axes
        plt.axis([0, self.iterations, 0, self.sme])
        plt.xlabel("Iteration")
        plt.ylabel("Squared Mean Error")
        #fig, ax = plt.subplots()
        #ax.set_ylabel("Squared Mean Error")
        #ax.set_xlabel("Iteration")
      plt.scatter(iteration, self.sme, color = 'black', s = 1)
      plt.pause(0.05) # plots real-time
      self.plot = True 

  def train(self):
    for i in range(self.iterations):
      self.time_start()
      self.iteration += 1
      self.feedforward()
      self.backprop()
      self.time_end()
      print("Iteration {} training time: {}".format(self.iteration, self.time))
      self.graph_sme(i)
    self.feedforward()
    self.confusion_matrix()
    if self.plot:
      plt.show()
