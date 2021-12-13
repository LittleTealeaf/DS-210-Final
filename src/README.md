[Importing Code File](data_import.py)
# Databases Used:
## Connect 4 Dataset
[Source](https://archive.ics.uci.edu/ml/datasets/Connect-4)

[Dataset](connect-4-data.csv)


# Soruce Code
This folder contains all of the source code, including resources and python files used in designing, testing, and teaching the neural network. A quick overview of each file is as follows:
*todo: put this in table*
*THESE LINKS DO NOT WORK*
- **[resources/](resources/)**: directory containing both datasets used in the project.
  - **[resources/connect-4-data.csv](./resources/connect-4-data.csv)**: Full dataset pulled from the [University California Irvine Database](https://archive.ics.uci.edu/ml/datasets/Connect-4). First row depicts headers for each column, and subsequent rows represent individual states of the board. More information can be found on the [resources/README.md](resources/README.md).
  - **[resources/connect-4-data-short.csv](./resources/connect-4-data.csv)**: Shortened dataset, contains only the first few rows of [connect-4-data.csv](resources/connect-4-data.csv) for use in testing, where the full dataset is not ideal to use
- **[data_import.py](data_import.py)**: Python script that contains methods to read data from the given CSV files **TODO: MAKE A METHOD TO GET EITHER FILE MANUALLY**
- **[lab.py](lab.py)**: Python script that contains the lab itself (probably needs a rename)
