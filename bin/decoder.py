import os
import struct
import numpy as np
from PIL import Image





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



# ---------------------- Example Usage ----------------------
if __name__ == '__main__':
    image_size = (8, 8)  # 8x8 image gives 8*8*3 = 192 bytes available.
    # Decode images back to matrices into the 'decoded_matrices' folder.
    decode_all_images(input_folder='encoded_matrices', output_folder='decoded_matrices', image_size=image_size)

