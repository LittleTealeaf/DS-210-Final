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

def backwards_propagation(Y2,W1_in,W0_in,S,y_actual):
    W1 = np.copy(W1_in,copy = True)
    W0 = np.copy(W0_in,copy = True)
    Y1 = sigmoid(W1 @ Y2)
    Y0 = sigmoid(W0 @ Y1)
    for i in range(len(W1)):
        deltaW = -1 * S * 2 * (Y0[0,0] - y_actual) * Y0[0,0] * (1 - Y0[0,0]) * Y1[i,0]
        W0[0,i] = W0[0,i] + deltaW
        for j in range(len(Y2)):
            W1[i,j] = W1[i,j] + deltaW * W0[0,i] * (1 - Y1[i,0]) * Y2[j,0]
    return (W1,W0)