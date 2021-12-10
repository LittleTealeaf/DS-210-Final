import numpy as np

# TODO: proper comments

def read_file(fileName):
    '''
    Reads the raw connect-4 data and converts it into a single matrix
    '''
    return np.loadtxt(fileName,dtype=float,delimiter=",",converters=generate_converters(43),skiprows=1)

def generate_converters(columnCount):
    '''
    Creates a generator that assigns each column number to use a lambda function that searches the string up on tile_dictionary
    '''
    converters = {}

    tile_dictionary = {
        "b": 0, "x": 1, "o": -1, "draw": 0, "win": 1, "loss": 0
    }

    for i in range(columnCount):
        converters[i] = lambda s: tile_dictionary[s.decode("utf-8")]
    return converters

def get_data(fileName):
    '''
    Returns two matricies, the first being the data, and the second being the expected outputs

    '''
    data_full = read_file(fileName)

    data_inputs = data_full[:,:-1]
    data_outputs = data_full[:,-1:]

    return data_inputs, data_outputs
