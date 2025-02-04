import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast




# Define the transforms:
# - Resize images to 128x128 (to match the image_size of the diffusion model)
# - Convert images to tensor (which scales pixel values to [0, 1])
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # Optionally, you could normalize your images here if needed.
])
print("finish resizing")
# Create a dataset.
# For ImageFolder, images should be arranged in subdirectories (one per class). 
# If you have only cats, you can either put them in one subfolder or adjust accordingly.
dataset = datasets.ImageFolder(root='./encoded_images', transform=transform)

# Adjust DataLoader for better performance
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)



model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False  # Disable flash attention
)


diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000    # number of steps
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
diffusion.to(device)
optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-5)



epochs = 1000  # Number of training epochs, adjust as needed.
print("starting epoch")

# Training loop without mixed precision
for epoch in range(epochs):
    for batch in dataloader:
        images, _ = batch
        images = images.to(device)
        
        optimizer.zero_grad()
        
        loss = diffusion(images)
        
        loss.backward()
        optimizer.step()
        # print("step")
    
    print(f"Epoch {epoch+1}/{epochs}: Loss = {loss.item()}")

# after a lot of training

sampled_images = diffusion.sample(batch_size=50)
print("Sampled images shape:", sampled_images.shape)  # Expected: (4, 3, 128, 128)

import os
from torchvision.utils import save_image

os.makedirs("generate_molecule_samples", exist_ok=True)


# Save each image separately
for i, image in enumerate(sampled_images):
    # save_image expects a tensor in shape (C, H, W)
    save_image(image, os.path.join("generated_molecule_samples", f"sample_{i}.png"))


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

def decode_all_images(input_folder='encoded_original_images', output_folder='decoded_original_matrices', image_size=(64, 64)):
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
            # print(f"Decoded matrix saved to {output_path}")

# Run the decoding process
decode_all_images(input_folder="generated_molecule_samples", output_folder="decoded_generated_molecules")
# To make training faster, you can use mixed precision training and DataLoader optimizations.


