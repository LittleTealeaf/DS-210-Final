import numpy as np

# TODO: proper comments

def read_file(fileName):
    '''
    Reads the raw connect-4 data and returns it as a single matrix (excluding the first row)
    
    Inputs:
     - fileName: file name (path) of .csv data
    Output: 2-dimensional matrix of data extracted from the CSV file, using generate_converters to convert letters into numbers
    '''
    return np.loadtxt(fileName,dtype=float,delimiter=",",converters=generate_converters(43),skiprows=1)

def generate_converters(columnCount):
    '''
    Creates a generator that assigns each column number to use a lambda function that searches the string up on tile_dictionary
    
    Input: number of columns to use the converter on
    Output: map from all specified columns to the respective dictionary to convert from raw data to numerical data
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

    return data_inputs, np.transpose(data_outputs)[0]

def get_data_full():
    return get_data("resources/connect-4-data.csv")

def get_data_short():
    return get_data("resources/connect-4-data-short.csv")
