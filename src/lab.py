import numpy as np
from data_import import *
import math
import random
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
            a[i] = sigmoid(n[i])
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
    W1 = np.copy(W1_in)
    W0 = np.copy(W0_in)
    Y1 = sigmoid(W1 @ Y2)
    Y0 = sigmoid(W0 @ Y1)
    Err = abs(Y0[0,0] - y_actual)
    for i in range(len(W1)):
        deltaW = -1 * S * 2 * (Y0[0,0] - y_actual) * Y0[0,0] * (1 - Y0[0,0]) * Y1[i,0]
        W0[0,i] = W0[0,i] + deltaW
        for j in range(len(Y2)):
            W1[i,j] = W1[i,j] + deltaW * W0[0,i] * (1 - Y1[i,0]) * Y2[j,0]
    return (W1,W0,Err)

def random_matrix(size):
    m = np.zeros(size)
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i,j] = random.uniform(0,1)
    return m


if __name__ == "__main__":
    in_data, out_data = get_data_full()
    # W1 = np.zeros((10,42))
    hidden_count = 10
    W1 = random_matrix((hidden_count,42))
    W0 = random_matrix((1,hidden_count))
    # W0 = np.zeros((1,10))
    total_error = 0
    iterations = 1000000
    for i in range(iterations):
        index = random.randrange(0,len(in_data))
        input = np.array([[j] for j in in_data[index]])
        output = out_data[index,0]
        W1,W0,error = backwards_propagation(input,W1,W0,1 + i * -(1.0 / iterations),output)
        total_error += error
        if(i%10000 == 0):
            print(total_error / len(in_data))
            total_error = 0
        if i%10000 == 0:
            print("actual:",output,"calculated:",sigmoid(W0 @ sigmoid(W1 @ input))[0,0])
    
