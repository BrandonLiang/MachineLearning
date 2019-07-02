#!/bin/python

import numpy as np

A = np.array([[6,1,1],
              [4, -2, 5],
              [2, 8, 7]])

print("Rank of A:", np.linalg.matrix_rank(A))
