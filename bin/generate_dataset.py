#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import sys
import re
from sklearn.metrics import f1_score, precision_score, recall_score


def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    df = pd.DataFrame(data.tolist() if data.dtype == 'O' and isinstance(data[0], dict) else data)
    return df


def npy_preprocessor(filename):
    df = read_data(filename)
    return df['index'].values, df['inchi'].values, df['xyz'].values, df['chiral_centers'].values, df['rotation'].values


def filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    if task == 0:
        filtered_indices = [i for i in range(len(index_array))]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array

    if task == 1:
        #return only chiral_length <2
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) < 2]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array

    elif task == 2:
        #only return chiral legnth < 5
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) < 5]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array
    elif task == 3: 
        # Step 1: Filter indices where the length of chiral_centers_array is exactly 1 and the first tuple contains 'R' or 'S'
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) == 1 and ('R' == chiral_centers_array[i][0][1] or 'S' == chiral_centers_array[i][0][1])]
        # Step 2: Create filtered arrays for index_array, xyz_arrays, chiral_centers_array, and rotation_array
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        
        # Step 5: Filter the rotation_array accordingly
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array
    
    elif task == 4:
        # only return chiral_length == 1
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) == 1]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array
    elif task == 5:
        filtered_indices = [i for i in range(len(index_array))]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array


def generate_label(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    # Task 0 or Task 1: Binary classification based on the presence of chiral centers
    if task == 0 or task == 1:
        return [1 if len(chiral_centers) > 0 else 0 for chiral_centers in chiral_centers_array]
    
    # Task 2: Return the number of chiral centers
    elif task == 2:
        return [len(chiral_centers) for chiral_centers in chiral_centers_array]
    
    # Task 3: Assuming that the task is to return something from chiral_centers_array, not rotation_array
    elif task == 3:
        return [
            1 if chiral_centers and len(chiral_centers[0]) > 1 and 'R' == chiral_centers[0][1] else 0
            for chiral_centers in chiral_centers_array
        ]

    
    # Task 4 or Task 5: Binary classification based on posneg value in rotation_array
    elif task == 4 or task == 5:
        return [1 if posneg[0] > 0 else 0 for posneg in rotation_array]

def generate_label_array(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    # Fix to directly return the output of generate_label
    return generate_label(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)



index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor('qm9_filtered.npy')


# In[2]:


import os


print(xyz_arrays[0].shape)

# print(xyz_arrays[0].shape) yields (27, 8), pad with zeros for (27, 12)
def pad_xyz(xyz_array):
    padded_array = np.zeros((xyz_array.shape[0], 12))
    padded_array[:, :xyz_array.shape[1]] = xyz_array
    return padded_array

# Example usage
padded_xyz_array = pad_xyz(xyz_arrays[0])
print(padded_xyz_array.shape)
print(padded_xyz_array)

# for all xyz_arrays pad
xyz_arrays = [pad_xyz(xyz) for xyz in xyz_arrays]


# Create the directory if it doesn't exist
os.makedirs('original', exist_ok=True)

# Save each padded xyz array to a txt file
for i, xyz in enumerate(xyz_arrays):
    np.savetxt(f'original/{index_array[i]}.txt', xyz)


# In[3]:


print(chiral_centers_array[0])


# In[4]:


# takes in a array tuple, the first item is the index, the second item is R, r, S, s, Tet_CCW, or Tet_CW encodes to 0,1,2,3,4,5 respectively create a array filled with encoded values at index 
# SAMPLE "[(0, 'S'), (1, 'R'), (2, 'r'), (3, 's'), (4, 'R'), (5, 'S')]"

def chiral_parser(chiral_array):
    if not chiral_array:
        return np.zeros(12, dtype=int)
    encoding_dict = {'R': 0, 'r': 1, 'S': 2, 's': 3, 'Tet_CCW': 4, 'Tet_CW': 5}
    encoded_array = np.zeros(12, dtype=int)
    
    for index, chiral in chiral_array:
        if chiral in encoding_dict:
            encoded_array[index] = encoding_dict[chiral]
    
    return encoded_array

# Example usage
sample_chiral_array = [(0, 'S'), (1, 'R'), (2, 'r'), (3, 's'), (4, 'R'), (5, 'S')]
encoded_values = chiral_parser(sample_chiral_array)
print(encoded_values)


# In[5]:


# print(xyz_arrays[0].shape) yields  (27, 12) append chiral_parser to make 28 x 12
def append_chirality(xyz_array, chiral_center_array):
    encoded_chirality = chiral_parser(chiral_center_array)
    # Reshape encoded_chirality to match the dimensions of xyz_array
    encoded_chirality = encoded_chirality.reshape(1, -1)
    # Append the encoded chirality to the xyz_array
    extended_xyz_array = np.vstack([xyz_array, encoded_chirality])
    return extended_xyz_array

# Example usage
extended_xyz_array = append_chirality(xyz_arrays[0], chiral_centers_array[0])
print(extended_xyz_array.shape)
print(extended_xyz_array)


# In[6]:


# for i in range len(index_array) create a encoded_molecule_matrix using the xyz_arrays[i] and chiral_parser(chiral_centers_arrays[i]) append_chirality(xyz_arrays[i], hiral_parser(chiral_centers_arrays[i])) return list of the encoded_molecules_matrix
encoded_molecules_matrix = []
for i in range(len(index_array)):
    encoded_molecule = append_chirality(xyz_arrays[i], chiral_centers_array[i])
    encoded_molecules_matrix.append(encoded_molecule)

print(len(encoded_molecules_matrix))


# In[7]:


import os
import struct
import numpy as np
from PIL import Image


def encode_matrix_to_image(matrix, output_file='index.png', image_size=(8, 8)):
    """
    Encodes a matrix of floats into an image using a lossless PNG format.
    Data format:
      - 1 byte: number of rows (assumes <= 255)
      - 1 byte: number of columns (assumes <= 255)
      - Then each float is stored as an 8-byte double (IEEE 754)
      - The remaining bytes (if any) are padded with 0.
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    # Pack rows and cols as unsigned bytes.
    data = struct.pack('BB', rows, cols)
    
    # Pack each float as an 8-byte double.
    for row in matrix:
        for value in row:
            data += struct.pack('d', value)
    
    # Calculate total capacity for an 8-bit RGB image.
    capacity = image_size[0] * image_size[1] * 3
    if len(data) > capacity:
        raise ValueError("Matrix data is too large for an image of size {}x{}".format(*image_size))
    
    # Pad the data.
    data_padded = data + b'\x00' * (capacity - len(data))
    
    # Convert to a numpy array and reshape.
    arr = np.frombuffer(data_padded, dtype=np.uint8).reshape((image_size[1], image_size[0], 3))
    
    # Create an image from the array.
    img = Image.fromarray(arr, 'RGB')
    
    # Save using PNG for lossless compression.
    img.save(output_file, format='PNG')
    print(f"Encoded matrix saved to {output_file}")


# Create the directory if it doesn't exist
os.makedirs('encoded_images', exist_ok=True)

# Encode each matrix and save to the encoded_images folder
for i in range(len(index_array)):
    encode_matrix_to_image(encoded_molecules_matrix[i], output_file=f'encoded_images/{index_array[i]}.png', image_size=(64, 64))


# In[5]:


import os
import struct
import numpy as np
from PIL import Image



def decode_image_to_matrix(image_path, image_size=(64, 64)):
    """
    Decodes an image (created by encode_matrix_to_image) back into the original matrix.
    """
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    data = arr.tobytes()
    
    # Unpack the first 2 bytes for rows and columns.
    rows, cols = struct.unpack('BB', data[:2])
    num_floats = rows * cols
    expected_length = 2 + num_floats * 8
    matrix_bytes = data[2:expected_length]
    
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            offset = (i * cols + j) * 8
            value = struct.unpack('d', matrix_bytes[offset:offset+8])[0]
            row.append(value)
        matrix.append(row)
    return np.array(matrix)

def decode_all_images(input_folder='encoded_images', output_folder='decoded_matrices', image_size=(64, 64)):
    """
    Decodes all images in the input folder and saves the resulting matrices in the output folder.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            matrix = decode_image_to_matrix(image_path, image_size=image_size)
            
            # Save the decoded matrix to a text file
            output_filename = filename.replace('.png', '.txt')
            output_path = os.path.join(output_folder, output_filename)
            np.savetxt(output_path, matrix)
            print(f"Decoded matrix saved to {output_path}")

# Run the decoding process
decode_all_images()

