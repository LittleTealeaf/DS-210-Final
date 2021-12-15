import numpy as np


def read_file(fileName: str):
    """Reads the raw connect-4 data from a file and converts it into a single matrix. Excludes the column headers (the first row) from reading

    Input:
    - fileName: Name of the file to read

    Output:
    - A matrix containing the raw data from the file.
    """
    return np.loadtxt(fileName,dtype=float,delimiter=",",converters=generate_converters(43),skiprows=1)

def generate_converters(columnCount: int = 43):
    """Creates a generator that assigns each column number to use a lambda function that replaces the string with an associated number.
    
    Input:
    - columCount (default = 43): The number of columns to use the converter on. With the connect-4 dataset, there are 43 columns which should be converted.
    
    Output:
    - Dictionary that associates each column provided to a lambda converter that looks up the string on the dictionary
    """
    converters = {}

    tile_dictionary = {
        "b": 0, "x": 1, "o": -1, "draw": 0, "win": 1, "loss": 0
    }

    converter = lambda s: tile_dictionary[s.decode("utf-8")]

    for i in range(columnCount):
        converters[i] = converter
    return converters



def get_data(fileName: str):
    """Reads data from a file, separating it into an input matrix and an output matrix.
    
    Input:
    - fileName: Name of the file to read.
    
    Outputs:
    - The input matrix, where each row contains the input data for that entry
    - The output matrix, where each entry returns the output vector expected for the corresponding input vector
    """
    data_matrix = read_file(fileName)
    return [np.array([[j] for j in i]) for i in data_matrix[:,:-1]], [np.array([[j] for j in i]) for i in data_matrix[:,-1:]]

def get_data_full():
    """Returns data from the full connect-4 data file, separating it into an input matrix and an output matrix.

    Outputs:
    - The input matrix, where each row contains the input data for that entry
    - The output matrix, where each entry returns the output vector expected for the corresponding input vector
    """
    return get_data("resources/connect-4-data.csv")

def get_data_short():
    """Returns data from the short connect-4 data file, separating it into an input matrix and an output matrix. The short conect-4 data is used for quick verification as it only contains a small list of entries.

    Outputs:
    - The input matrix, where each row contains the input data for that entry
    - The output matrix, where each entry returns the output vector expected for the corresponding input vector
    """
    return get_data("resources/connect-4-data-short.csv")
