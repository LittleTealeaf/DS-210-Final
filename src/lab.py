import numpy as np
import math
import random
from data_import import *
"""
Stochastic Descent and Back Propagation Lab
Author: Thomas Kwashnak
"""


def sigmoid(n):
    """Calculate the sigmoid function of a given input.
    Input:
        n - Either a scalar value, or a single or multidimensional array
    Output:
        m - The value of n passed through the sigmoid function. If n is an array, then each value within n is passed through the sigmoid function
    """
    if isinstance(n,np.ndarray):
        a = np.zeros(n.shape,dtype=n.dtype)
        for i in range(len(a)):
            a[i] = sigmoid(n[i])
        return a
    else:
        return 1 / (1 + math.exp(-n))

def feed_forward(Y,W):
    """Feeds forward the values from a previous layer to the next layer closest to the output layer
    
    Inputs:
    - Y: The previous layer values, as a matrix of size h x 1.
    - W: The weight matrix, as a matrix of size k x h, where k is the number of nodes in the target layer
    
    Output:
     - The resulting values of the nodes at the target layer"""
    return sigmoid(W @ Y)

def back_propagation(Y2: np.matrix, W1: np.matrix, W0: np.matrix, y_expected: np.double):
    """Back Propagation for a 2-layer neural network
    
    Inputs:
    - Y2: The input vector (matrix), dimensions m x 1
    - W1: The weight matrix, dimensions h x m, that represents the weights used as values go from Y2 to Y1
    - W0: The weight matrix, dimensions 1 x h, that represents the weights used as values go from Y1 to Y0
    - y_expected: The expected result from Forward-Feeding

    Outputs:
    - A matrix of the derivatives of the weights in W1, as a h x m matrix
    - A matrix of the derivatives of the weights in W0, as a 1 x h matrix
    - The absolute error of the network for the given input
    """
    dW1 = np.zeros(W1.shape)
    dW0 = np.zeros(W0.shape)
    Y1 = feed_forward(Y2,W1)
    Y0 = feed_forward(Y1,W0)
    Err = abs(Y0[0,0] - y_expected)
    for i in range(len(W1)):
        dW0[0,i] = 2 * (Y0[0,0] - y_expected ) * Y0[0,0] * (1 - Y0[0,0]) * Y1[i,0]
        for j in range(len(Y2)):
            dW1[i,j] = dW0[0,i] * W0[0,i] * (1 - Y1[i,0]) * Y2[j,0]
    return dW1,dW0,Err

def stochastic_descent(inputs: list,outputs: list,W1: np.matrix,W0: np.matrix,step: np.double):
    """Stochastic Gradient Descent Iteration
    
    Inputs:
    - inputs: length n list of input vectors, as matrices. Each input vector is of size m x 1
    - output: length n list of all expected values, as scalars. Each output corresponsds to the input at the same index.
    - W1: The weight matrix, dimensions h x m, that represents the weights used as values go from Y2 to Y1
    - W0: The weight matrix, dimensions 1 x h, that represents the weights used as values go from Y1 to Y2
    - step: The step coefficient to nudge the weights by
    
    Outputs:
    - The updated weight matrix W1, of dimensions h x m, where values have been modified by the average derivative across all inputs/outputs scaled by the step
    - The updated weight matrix W0, of dimensions 1 x h, where values have been modified by the average derivative across all inputs/outputs scaled by the step
    - The average absolute error of the network on each input
    """
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

def train_network(inputs,outputs,W1_in,W0_in,f_step,iteration_count,batch_size,seed=None):
    """Trains the neural network on a set of inputs, outputs, and values.

    Inputs:
    - inputs: length n list of all input vectors, as matrices. Each input vector is of size m x 1
    - outputs: length n list of all expected values, as scalars. Each output corresponds to the input at that same index.
    - W1_in: The weight matrix, of dimensions h x m, that represents the weights used as values go from Y2 to Y1.
    - W0_in: The weight matrix, of dimensions 1 x h, that represents teh weights used as values go from Y1 to Y0.
    - f_step: A lambda function that correlates a given iteration i to a step value.
    - iteration_count: The number of iterations to run the network on.
    - batch_size: The number of entries to batch together for each iteration
    - seed: The seed to base the randomization off of

    Outputs:
    - The updated weight matrix W1, dimensions h x m, trained on the given inputs and outputs.
    - The updated weight matrix W0, dimensions 1 x h, trained on the given inputs and outputs
    
    """
    W1 = W1_in
    W0 = W0_in
    if seed:
        random.seed(seed)
    for i in range(iteration_count):
        step = f_step(i)
        indexes = [random.randint(0,len(inputs)-1) for j in range(batch_size)]
        input_iteration = [inputs[j] for j in indexes]
        output_iteration = [outputs[j] for j in indexes]
        W1,W0,Err = stochastic_descent(input_iteration,output_iteration,W1,W0,step)
        # print("Error:",Err)
    return W1,W0

def evaluate_network(inputs,outputs,W1,W0):
    """Evaluates the network based on the average absolute error across all entries.
    
    Inputs:
    - inputs: length n list of all input vectors, as matrices. Each input vector is of size m x 1
    - outputs: length n list of all expected values, as scalars. Each output corresponds to the input at that same index.
    - W1: The weight matrix, of dimensions h x m, that represents the weights used as values go from Y2 to Y1.
    - W0: The weight matrix, of dimensions 1 x h, that represents teh weights used as values go from Y1 to Y0.
    
    Outputs:
    - Average absolute error across all inputs"""
    total_error = 0
    for i in range(len(inputs)):
        Y0 = sigmoid(W0 @ sigmoid(W1 @ inputs[i]))
        total_error += abs(Y0[0,0] - outputs[i])
    return total_error[0,0] / len(inputs)

def random_matrix(size,seed: int=None):
    """Creates a matrix from a give size, comprised of random values betwen 0 and 1
    
    Inputs:
    - size: tuple of dimensions for the matrix
    - seed (default = random): the seed to begin at"""
    if seed:
        np.random.seed(seed)
    return np.random.random_sample(size)

def perform_experiment():
    """
    Basically, an experimental procedure that I used in order to try to determine optimal settings I could use for my network. The results, as listed in the report, is that 19 nodes in the hidden layer, and a batch size of 3, tend to have the best results.
    """
    class Experiment:
        """A class that basically stores all the modifyable parameters of a given training, such that they can be tracked. This class is primarily used in running an automated experiment to find the optimal number of nodes in the hidden layer, iterations, and batch size."""
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
                W1,W0 = train_network(self.inputs,self.outputs,self.W1,self.W0,lambda n: 1.0 - n / self.iterations, self.iterations, self.batch_size, self.seed)
                self.evaluation = evaluate_network(self.inputs, self.outputs, W1, W0)
            return self.evaluation

    in_data, out_data = get_data_full()
    
    seed = 31415926

    iterations = 1000

    best = Experiment(in_data,out_data,seed,23,1000,1)

    for iterations in range(1000,10000,1000):
        for hidden_count in range(1,40,1):
            for batch_size in range(1,max(2 * best.batch_size,30),1):
                a = Experiment(in_data,out_data,seed,hidden_count,iterations,batch_size)
                if (best == None) or (best.get_evaluation() > a.get_evaluation()):
                    best = a
                print("Tested:",a.iterations,a.hidden_count,a.batch_size,a.get_evaluation()," \tBest:",best.iterations,best.hidden_count,best.batch_size,best.get_evaluation())
            
    # prev best: Best: 1000 23 1 5.404723143044208e-06
    print(a.hidden_count, a.iterations, a.batch_size)


def verification():
    """
    Runs verifications as seen in verification section of report
    """
    print("Verification Process:")
    print("Testing Sigmoid Function:")
    print("\tsigmoid(5):","Actual:",0.9933071490757153,"Found:",sigmoid(5))
    print("\tsigmoid([5,10]):","Actual:",[0.9933071490757153,0.9999546021312976],"Found:",sigmoid(np.array([5.,10.])))

    print("Testing Feed-Forward")
    Y2 = np.array([[1.],[-1.]])
    W1 = np.array([[1,0.5],[-1,0]])
    W0 = np.array([[0.5,2]])
    print("\tfeed_forward(Y2,W1)","Actual:",np.array([[0.6224593312018546],[0.2689414213699951]]),"Found:",feed_forward(Y2,W1))

    print("Backwards Propagation:")
    dW1,dW0,err = back_propagation(Y2,W1,W0,0.5)
    print("Expected dW1:")
    print(np.array([[0.009881773269086033,-0.009881773269086033],[.0330696826545844,-.0330696826545844]]))
    print("Found dW1:")
    print(dW1)
    print("Expected dW0:")
    print(np.array([[0.05234812610012824,0.02261766951463492]]))
    print("Found dW0:")
    print(dW0)
    print("Expected Error:",0.2003809377121779)
    print("Found Error:",err)

def train_new_network():
    """Main method used to train a new network with specified values below. Edit and experiment as needed:"""

    in_data,out_data = get_data_full()

    batch_size = 3
    hidden_count = 19
    iterations = 1000000
    evaluation_interval = 10000

    W1 = random_matrix((hidden_count,42),12)
    W0 = random_matrix((1,hidden_count),423)

    for i in range(iterations // evaluation_interval):
        W1,W0 = train_network(in_data,out_data,W1,W0,lambda n: 1.0 - (n + i * evaluation_interval) / iterations,evaluation_interval,batch_size)
        print("Evaluation at : ",(i * evaluation_interval),"Average Absolute Error:",evaluate_network(in_data,out_data,W1,W0))

    


if __name__ == "__main__":
    print("Author: Thomas Kwashnak")
    # perform_experiment()
    # verification()
    train_new_network()
    
