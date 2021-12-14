import numpy as np
import data_import
import math
from functools import lru_cache


def sigmoid(n):
    '''
    Input: A matrix, or a number, `n`
    
    Output: `signmoid(n)`. If `n` was a matrix, then will return a 
    new matrix of identical dimensions where each value is put into 
    the sigmoid function
    '''
    if isinstance(n,np.ndarray):
        a = np.zeros(n.shape)
        for i in range(len(a)):
            for j in range(len(a[i])):
                a[i,j] = sigmoid(n[i,j])
        return a
    else:
        return 1 / (1 + math.exp(-n))

def sigmoidDerivative(n):
    '''
    Input: A number `n`
    Output: The derivative of `sigmoid(n)` at that point
    '''
    return sigmoid(n) * (1 - sigmoid(n))


def forward_feeding(yIn, W):
    '''
    Feeds forwards a the input vector to the output

    Input: yIn - A h-length vector of the initial input values
    Input: W - An array or list of N Weight-Matrices of varying sizes. The index of the matrix should relate to the order at which they are closest to the output layer. The size of Wa should be i x j, where i is the number of nodes in that layer, and j is the number of nodes for the layer Wa+1. The last Weight-Matrix should be of size i x h

    Output: Y - The resulting vector of values outputted by the neural network
    '''
    Y = np.copy(yIn,copy=True)
    for i in range(len(W) - 1, -1, -1):
        Y = sigmoid(W[i] @ Y)
    return Y