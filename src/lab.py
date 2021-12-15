from typing import Iterator
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

def back_propagation(Y2, W1, W0, y_expected):
    dW1 = np.zeros(W1.shape)
    dW0 = np.zeros(W0.shape)
    Y1 = sigmoid(W1 @ Y2)
    Y0 = sigmoid(W0 @ Y1)
    Err = abs(Y0[0,0] - y_expected)
    for i in range(len(W1)):
        dW0[0,i] = 2 * (Y0[0,0] - y_expected) * Y0[0,0] * (1 - Y0[0,0]) * Y1[i,0]
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

def train_network(inputs,outputs,W1_in,W0_in,f_step,n,c,seed):
    W1 = W1_in
    W0 = W0_in
    random.seed(seed)
    for i in range(n):
        step = f_step(i,n)
        indexes = [random.randint(0,len(inputs)-1) for j in range(c)]
        input_iteration = [inputs[j] for j in indexes]
        output_iteration = [outputs[j] for j in indexes]
        W1,W0,Err = stochastic_descent(input_iteration,output_iteration,W1,W0,step)
        # print("Error:",Err)
    return W1,W0

def evaluate_network(inputs,outputs,W1,W0):
    total_error = 0
    for i in range(len(inputs)):
        Y0 = sigmoid(W0 @ sigmoid(W1 @ inputs[i]))
        total_error = abs(Y0[0,0] - outputs[i])
    return total_error / len(inputs)

def random_matrix(size):
    m = np.zeros(size)
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i,j] = random.uniform(0,1)
    return m

class Experiment:
    def __init__(self,inputs,outputs,seed,hidden_count,iterations,batch_size):
        self.inputs = inputs
        self.outputs = outputs
        self.seed = seed
        self.hidden_count = hidden_count
        self.iterations = iterations
        self.batch_size = batch_size
        self.evaluation = -1
        random.seed(seed)
        self.W1 = random_matrix((hidden_count,42))
        self.W0 = random_matrix((1,hidden_count))

    def get_evaluation(self):
        if(self.evaluation == -1):
            W1,W0 = train_network(self.inputs,self.outputs,self.W1,self.W0,lambda n,i: 1.0 - n / i, self.iterations, self.batch_size, self.seed)
            self.evaluation = evaluate_network(self.inputs, self.outputs, W1, W0)
        return self.evaluation


if __name__ == "__main__":
    in_data, out_data = get_data_full()
    in_data = [np.array([[j] for j in i]) for i in in_data]
    
    seed = 31415926

    iterations = 1000

    best = Experiment(in_data,out_data,seed,1,iterations,20)

    for iterations in range(1000,10000,500):
        for hidden_count in range(1,30,1):
            for batch_size in range(1,max(2 * best.batch_size,50),1):
                if best != None:
                    print("Testing: ",best.iterations,hidden_count,batch_size,"Best:",best.iterations,best.hidden_count,best.batch_size,best.get_evaluation())
                a = Experiment(in_data,out_data,seed,hidden_count,iterations,batch_size)
                if (best == None) or (best.get_evaluation() > a.get_evaluation()):
                    best = a
            
    print(a.hidden_count, a.iterations, a.batch_size)

    # hidden_count = 25
    # iterations = 10000
    # batch_size = 5
    # seed = 32156


    # W1 = random_matrix((hidden_count,42))
    # W0 = random_matrix((1,hidden_count))
    # W1,W0 = train_network(in_data,out_data,W1,W0,lambda n,i: 1.0 - n / i, iterations,batch_size,seed)
    # # print(W1)
    # # print(W0)
    # print("Evlatuation:",evaluate_network(in_data,out_data,W1,W0))


