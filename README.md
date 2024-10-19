# KNN-Transformer Model

This repository implements a deep learning model combining K-Nearest Neighbor (KNN) attention with transformers for voxel data classification tasks. The model preprocesses voxel data, trains on it, and performs evaluation and visualization of results.

## Features
- Preprocess voxel data with different classification tasks.
- Utilize K-Nearest Neighbor attention to enhance feature extraction based on 3D spatial coordinates.
- Train a transformer model with KNN-based attention and positional encoding.
- Evaluate model performance using F1 score and confusion matrix.
- Visualize model predictions and confusion matrix.

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
git clone https://github.com/aqjshi/KNN-Transformer.git
cd KNN-Transformer
pip install -r requirements.txt

```

## Run KNN Transformer
python KNN_Transformer.py

## Run FFNN 
python FFNN.py





