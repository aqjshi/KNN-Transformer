from dataset import npy_preprocessor, rotate_xyz, augment_dataset, evaluate_with_f1
from KNN import TransformerModelWithKNN, KNearestNeighborAttention
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch.nn.functional as F


# Evaluation function that calculates loss, accuracy, and F1 score
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Generate binary classification label (based on rotation array)
def generate_label(chiral_center):
    return 1 if chiral_center > 0 else 0

# Load and process data
index_array, inchi_array, xyz_array, chiral_centers_array, rotation_array = npy_preprocessor('bin2/qm9_filtered.npy')

# Convert xyz_array to numeric data (float32)
xyz_array = np.array([np.array(x) for x in xyz_array], dtype=np.float32)

# Generate labels (binary classification based on chiral centers)
label_array = [generate_label(chiral_centers[0]) for chiral_centers in rotation_array]

# Split the original data before augmentation
train_size = int(0.2 * len(xyz_array))  
test_size = len(xyz_array) - train_size
train_xyz, test_xyz, train_labels, test_labels = train_test_split(xyz_array, label_array, test_size=test_size, train_size=train_size)

# Augment only the training set (not the test set)
augmented_train_xyz, augmented_train_labels = augment_dataset(train_xyz, train_labels)

# Convert augmented train data and unmodified test data to PyTorch tensors
train_data_tensor = torch.tensor(augmented_train_xyz, dtype=torch.float32).to(device)  # Move to device
train_label_tensor = torch.tensor(augmented_train_labels, dtype=torch.float32).unsqueeze(1).to(device)  # Move to device

test_data_tensor = torch.tensor(test_xyz, dtype=torch.float32).to(device)  # Move to device
test_label_tensor = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1).to(device)  # Move to device

# Create TensorDatasets for train and test sets
train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
test_dataset = TensorDataset(test_data_tensor, test_label_tensor)

# DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


model = TransformerModelWithKNN(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=1, k_neighbors=5).to(device)

# Initialize loss function and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_train_labels = []
    all_train_predictions = []

    for batch in train_loader:
        data, labels = batch
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Collect predictions for F1 calculation
        predicted = (outputs > 0.5).float()
        all_train_predictions.extend(predicted.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

    avg_train_loss = running_loss / len(train_loader)
    train_f1 = f1_score(all_train_labels, all_train_predictions, average='weighted')
        
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}, Training F1 Score: {train_f1:.4f}")
    
    # Evaluate on test set and report F1 score
    test_loss, test_accuracy, test_f1 = evaluate_with_f1(model, test_loader, criterion)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test F1 Score: {test_f1:.4f}")
