import os
import numpy as np

# Directory to save the matrices
directory = '/Users/anthonys/Desktop/encoder/matrices'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
    # Generate 1000 3x3 matrices of random numbers between -1 and 10
    for i in range(1000):
        matrix = np.random.uniform(-1, 10, (8, 27))
        file_path = os.path.join(directory, f'matrix_{i+1}.txt')
        np.savetxt(file_path, matrix, fmt='%.10f')
    print("done")
