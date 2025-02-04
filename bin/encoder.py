import os
import struct
import numpy as np
from PIL import Image

def encode_matrix_to_image(matrix, output_file='encoded_matrix.png', image_size=(8, 8)):
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

def decode_image_to_matrix(image_path, image_size=(8, 8)):
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
    return matrix

def encode_all_matrices(input_folder='matrices', output_folder='encoded_matrices', image_size=(8,8)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.startswith('matrix_') and filename.endswith('.txt'):
            index = filename.split('_')[1].split('.')[0]
            with open(os.path.join(input_folder, filename), 'r') as f:
                matrix = [list(map(float, line.split())) for line in f if line.strip()]
            output_file = os.path.join(output_folder, f'encoded_matrix_{index}.png')
            encode_matrix_to_image(matrix, output_file, image_size)

def decode_all_images(input_folder='encoded_matrices', output_folder='decoded_matrices', image_size=(8,8)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.startswith('encoded_matrix_') and filename.endswith('.png'):
            index = filename.split('_')[2].split('.')[0]
            image_path = os.path.join(input_folder, filename)
            matrix = decode_image_to_matrix(image_path, image_size)
            output_file = os.path.join(output_folder, f'decoded_matrix_{index}.txt')
            with open(output_file, 'w') as f:
                for row in matrix:
                    f.write(' '.join(f'{x:.6g}' for x in row) + '\n')
            print(f"Decoded matrix saved to {output_file}")

# ---------------------- Example Usage ----------------------
if __name__ == '__main__':
    image_size = (64, 64)  # 8x8 image gives 8*8*3 = 192 bytes available.
    # Encode matrices from the 'matrices' folder.
    encode_all_matrices(input_folder='matrices', output_folder='encoded_matrices', image_size=image_size)
    # Decode images back to matrices into the 'decoded_matrices' folder.
    #decode_all_images(input_folder='encoded_matrices', output_folder='decoded_matrices', image_size=image_size)
    
