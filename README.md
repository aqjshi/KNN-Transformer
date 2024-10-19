# Voxel Transformer Model

This repository implements a deep learning model using transformers for voxel data classification tasks. The model preprocesses voxel data, trains on it, and performs evaluation and visualization of results.

## Features
- Preprocess voxel data with different classification tasks.
- Train a transformer model with patch embeddings and positional encoding.
- Evaluate model performance using F1 score and confusion matrix.
- Visualize model predictions and confusion matrix.

## Directory Structure
root/  

├──train_rs/ # Training data directory  

├── test_rs/ # Testing data directory  

├── main.py # Main script for training and evaluation  

├── model_predictions.png # Output image of model predictions  

└── confusion_matrix.png # Output image of confusion matrix  


## Classification Tasks
You can choose from the following classification tasks:
1. **Task 0:** Cast chiral_length > 0 to 1, keep 0.
2. **Task 1:** 0 vs. 1 chiral center.
3. **Task 2:** Number of chiral centers.
4. **Task 3:** R vs. S chirality.
5. **Task 4:** + vs. - chirality for chiral_length == 1.
6. **Task 5:** + vs. - chirality for all chiral lengths.

## Prerequisites
Make sure you have the following dependencies installed. You can install them using the provided `requirements.txt` file.

## Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/yourusername/voxel-transformer-model.git
cd voxel-transformer-model
pip install -r requirements.txt
```

## Generate Dataset
python cache_tensor_v2.py qm9_filtered.npy [Voxel Resolution: RECOMMEND 9]

## Run Spatial Transformer
python voxel_vit_v3.py






