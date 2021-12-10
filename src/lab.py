import numpy as np
import data_import
import math
from functools import lru_cache

def forward_feeding(In,W0,W1):
    '''
    Inputs:
    In - Input Matrix (42 x 1)
    W0 - 0th layer weights (closest to output) (1 x b)
    W1 - 1st layer weights (b x 42)

    Outputs:
     - The resulting chance that player 1 is winning
    '''
    return sigmoid(W0 @ sigmoid(W1 @ In))
    

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


