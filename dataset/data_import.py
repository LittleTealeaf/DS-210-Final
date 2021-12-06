import numpy as np

tile_dictionary = {
    "b": 0, "x": 1, "o": -1, "draw": 0, "win": 1, "loss": -1
}

'''
Goals of this script:
 - Read data from dataset into two matricies. One matrix contains the list of all inputs and another matrix has the proper outputs (p1 win, draw, p2 win)
'''


def read_file(fileName):
    '''
    Reads the raw connect-4 data and converts it into a single matrix
    '''
    return np.loadtxt(fileName,dtype=float,delimiter=",",converters=generate_converters(43))

def generate_converters(columnCount):
    converters = {}
    for i in range(columnCount):
        converters[i] = lambda s: tile_dictionary[s.decode("utf-8")]
    return converters

def get_data(fileName):
    '''
    Returns two matricies, the first being the data, and the second being the expected outputs
    '''
    data_full = read_file(fileName)

    data_inputs = data_full[:,:-1]

    data_outputs_partial = data_full[:,-1:]
    data_outputs = np.zeros((len(data_outputs_partial),3))
    for i in range(len(data_outputs)):
        data_outputs[i][int(data_outputs_partial[i][0])] = 1
    
    return data_inputs, data_outputs


print(get_data("connect-4-data.csv"))