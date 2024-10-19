from dataset import npy_preprocessor, rotate_xyz, augment_dataset, evaluate_with_f1, filter_data, generate_label, generate_label_array
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch.nn.functional as F


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        # Input to hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Hidden to output layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Pass through the first fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        # Pass through the second fully connected layer
        x = self.fc2(x)
        return x


task = 3
# Load and process data
index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor('bin2/qm9_filtered.npy')

index_array, xyz_arrays, chiral_centers_array, rotation_array = filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)
# Convert xyz_array to numeric data (float32)
xyz_array = np.array([np.array(x) for x in xyz_arrays], dtype=np.float32)

# Flatten xyz array to 1D
xyz_array = xyz_array.reshape(xyz_array.shape[0], -1)

# Generate binary classification labels
label_array = generate_label_array(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)
# Convert labels to a tensor
label_array = torch.tensor(label_array, dtype=torch.float32)

# Split the original data before augmentation
train_size = int(0.2 * len(xyz_array))  
test_size = len(xyz_array) - train_size
train_xyz, test_xyz, train_labels, test_labels = train_test_split(xyz_array, label_array, test_size=test_size, train_size=train_size)

# Convert to tensors
train_xyz_tensor = torch.tensor(train_xyz, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
test_xyz_tensor = torch.tensor(test_xyz, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

# Create DataLoader for training
train_dataset = TensorDataset(train_xyz_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Hyperparameters
input_size = train_xyz.shape[1]  # Number of features
hidden_size = 512  # Number of neurons in the hidden layer
output_size = 1  # Binary classification
learning_rate = 0.0005
num_epochs = 50

# Initialize the neural network
model = FFNN(input_size, hidden_size, output_size).to(device)



criterion = nn.BCEWithLogitsLoss()
# Cross entropy loss
if task == 2:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs.squeeze(), target)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if (epoch+1) % 5 == 0:
        # Evaluation on test data
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_xyz_tensor.to(device)).squeeze()
            test_preds = torch.sigmoid(test_outputs).round().cpu().numpy()
            f1 = f1_score(test_labels, test_preds)
            print(f'Test F1 Score: {f1:.4f}')


