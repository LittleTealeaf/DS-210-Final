import numpy as np
import math
import random
from data_import import *


def sigmoid(n):
    """Calculate the sigmoid function of a given input.
    Input:
        n - Either a scalar value, or a single or multidimensional array
    Output:
        m - The value of n passed through the sigmoid function. If n is an array, then each value within n is passed through the sigmoid function
    """
    if isinstance(n,np.ndarray):
        a = np.zeros(n.shape)
        for i in range(len(a)):
            a[i] = sigmoid(n[i])
        return a
    else:
        return 1 / (1 + math.exp(-n))

def sigmoid_derivative(n):
    """Calculates the derivative of the sigmoid function at a given input.
    Input:
        n - Either a scalar value, or a single or multidimensional array
    Output:
        m - The value of n passed through the derivative of the sigmoid function. If n is an array, then each value within n is passed through the derivative of the sigmoid function
    """
    return sigmoid(n) * (1 - sigmoid(n))

# def backwards_propagation(Y2,W1_in,W0_in,S,y_actual):
#     """Calculates and returns updated weights after updating to minimize the loss function from a given input.
#     Inputs:
#         Y2 - The input vector, dimensions m x 1
#         W1_in - The weight matrix, dimensions h x m, that represent the weights applied from the input layer to the 1st layer
#         W0_in - The weight matrix, dimensions 1 x h, that represent the weights applied from the 1st layer to the 0st layer (output)
#         S - The step coefficient, as a scalar, to modify how much change is made to the weights
#         y_actual - The expected value of the neural network, as a scalar value between [0,1]
#     Outputs:
#         W^1 - The updated weights, dimensions h x m, that represent the weights applied from the input layer to the 1st layer
#         W^0 - The updated weights, dimensions h x m, that represent the weights applied from the 1st layer to the 0st layer (output)
#         Err - The absolute error, as a scalar value, of the predicted output and the actual output for the given input
#     """
#     W1 = np.copy(W1_in)
#     W0 = np.copy(W0_in)
#     Y1 = sigmoid(W1 @ Y2)
#     Y0 = sigmoid(W0 @ Y1)
#     Err = abs(Y0[0,0] - y_actual)
#     for i in range(len(W1)):
#         deltaW = -1 * S * 2 * (Y0[0,0] - y_actual) * Y0[0,0] * (1 - Y0[0,0]) * Y1[i,0]
#         W0[0,i] = W0[0,i] + deltaW
#         for j in range(len(Y2)):
#             W1[i,j] = W1[i,j] + deltaW * W0[0,i] * (1 - Y1[i,0]) * Y2[j,0]
#     return (W1,W0,Err)

def back_propagation(Y2, W1, W0, y_expected):
    dW1 = np.zeros(W1.shape)
    dW0 = np.zeros(W0.shape)
    Y1 = sigmoid(W1 @ Y2)
    Y0 = sigmoid(W0 @ Y1)
    Err = abs(Y0[0,0] - y_expected)
    for i in range(len(W1)):
        dW0[0,i] = 2 * (y_expected - Y0[0,0]) * Y0[0,0] * (1 - Y0[0,0]) * Y1[i,0]
        for j in range(len(Y2)):
            dW1[i,j] = dW0[0,i] * W0[0,i] * (1 - Y1[i,0]) * Y2[j,0]
    return dW1,dW0,Err

def stochastic_descent(inputs,outputs,W1,W0,step):
    Err_total = 0
    DW1 = np.zeros(W1.shape)
    DW0 = np.zeros(W0.shape)
    for i in range(len(inputs)):
        dW1,dW0,dErr = back_propagation(inputs[i],W1,W0,outputs[i])
        DW1 = DW1 + dW1
        DW0 = DW0 + dW0
        Err_total = Err_total + dErr
    DW1 = DW1 
    DW0 = DW0
    Err_total = Err_total / len(inputs)
    W1 = W1 - DW1 * step / len(inputs)
    W0 = W0 - DW0 * step / len(inputs)
    return W1, W0, Err_total

def train_network(inputs,outputs,W1,W0,f_step,n,c,seed):
    random.seed(seed)
    for i in range(n):
        step = f_step(i,n)
        indexes = [random.randint(0,len(inputs)-1) for j in range(c)]
        input_iteration = [inputs[j] for j in indexes]
        output_iteration = [outputs[j] for j in indexes]
        W1,W0,Err = stochastic_descent(input_iteration,output_iteration,W1,W0,step)
        print("Error:",Err)
    return W1,W0

def random_matrix(size):
    m = np.zeros(size)
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i,j] = random.uniform(0,1)
    return m


if __name__ == "__main__":
    in_data, out_data = get_data_full()
    in_data = [np.array([[j] for j in i]) for i in in_data]
    hidden_count = 5
    W1 = random_matrix((hidden_count,42))
    W0 = random_matrix((1,hidden_count))
    train_network(in_data,out_data,W1,W0,lambda n,i: 1.0 - n / i, 100,100000,1111)
