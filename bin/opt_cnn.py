import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# CNN Model
class CNN(nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU()
        )
        flatten_size = 128 * 13 * 4
        self.fc_layer = nn.Sequential(nn.Linear(flatten_size, output_size))

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# Data preprocessing functions (placeholders)
def npy_preprocessor(filename):
    # Replace with the actual implementation for loading and preprocessing your dataset
    pass

def filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    # Replace with the actual filtering logic based on the task
    pass

def split_data(index_array, xyz_arrays, chiral_centers_array, rotation_array):
    train_idx, test_idx = train_test_split(range(len(index_array)), test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.05, random_state=42)

    def subset(indices):
        return ([index_array[i] for i in indices],
                [xyz_arrays[i] for i in indices],
                [chiral_centers_array[i] for i in indices],
                [rotation_array[i] for i in indices])
    return subset(train_idx), subset(val_idx), subset(test_idx)

def normalize_xyz_train(xyz_arrays):
    all_xyz = np.concatenate([xyz[:, :3] for xyz in xyz_arrays], axis=0)
    min_val = all_xyz.min()
    max_val = all_xyz.max()
    norm_xyz = [(xyz - min_val) / (max_val - min_val) for xyz in xyz_arrays]
    return min_val, max_val, norm_xyz

def apply_normalization(xyz_arrays, min_val, max_val):
    return [(xyz - min_val) / (max_val - min_val) for xyz in xyz_arrays]

def generate_label(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    # Placeholder: Replace with logic for generating task-specific labels
    pass

# Training function
def train_model(train_data, val_data, test_data, task, num_epochs, batch_size, learning_rate, l2_lambda):
    train_idx, train_xyz, train_chiral, train_rot = train_data
    val_idx, val_xyz, val_chiral, val_rot = val_data

    min_val, max_val, train_xyz = normalize_xyz_train(train_xyz)
    val_xyz = apply_normalization(val_xyz, min_val, max_val)

    train_labels_np = np.array(generate_label(train_idx, train_xyz, train_chiral, train_rot, task))
    val_labels_np = np.array(generate_label(val_idx, val_xyz, val_chiral, val_rot, task))

    train_tensor = torch.tensor(np.array(train_xyz), dtype=torch.float32)
    val_tensor = torch.tensor(np.array(val_xyz), dtype=torch.float32)

    train_labels = torch.tensor(train_labels_np, dtype=torch.float32)
    val_labels = torch.tensor(val_labels_np, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor, train_labels)
    val_dataset = TensorDataset(val_tensor, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    output_size = 1
    model = CNN(output_size).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs).squeeze()
            loss = criterion(outputs, labels) + weight_decay(model, l2_lambda, device)
            loss.backward()
            optimizer.step()

        # Evaluate on validation data (optional for logging purposes)
        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, task, test=False)

    return model

# Evaluation function
def evaluate_model(model, loader, criterion, task, test=False):
    model.eval()
    all_labels, all_predictions = [], []
    running_loss = 0.0

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            data = data.unsqueeze(1)
            outputs = model(data)
            outputs = torch.sigmoid(outputs).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    f1 = f1_score(all_labels, all_predictions, average='binary')
    return avg_loss, accuracy, None, None, f1

# Weight decay calculation
def weight_decay(cnn_model, l2_lambda, device):
    reg = 0.0
    for param in cnn_model.parameters():
        reg += torch.norm(param, 2)**2
    return (l2_lambda / 2) * reg

# Optuna objective function
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    l2_lambda = trial.suggest_loguniform('l2_lambda', 1e-6, 1e-2)
    num_epochs = trial.suggest_int('num_epochs', 10, 50)

    index_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor('dataset.npy')
    train_data, val_data, test_data = split_data(index_array, xyz_arrays, chiral_centers_array, rotation_array)
    model = train_model(train_data, val_data, test_data, task=0, num_epochs=num_epochs, batch_size=batch_size,
                        learning_rate=learning_rate, l2_lambda=l2_lambda)

    val_idx, val_xyz, val_chiral, val_rot = val_data
    val_labels_np = np.array(generate_label(val_idx, val_xyz, val_chiral, val_rot, task=0))
    val_tensor = torch.tensor(np.array(val_xyz), dtype=torch.float32)
    val_labels = torch.tensor(val_labels_np, dtype=torch.float32)
    val_dataset = TensorDataset(val_tensor, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    _, _, _, _, val_f1 = evaluate_model(model, val_loader, criterion, task=0, test=False)

    return -val_f1  # Maximize F1 by minimizing its negative

# Run Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best trial:")
print(f"Value (F1): {-study.best_trial.value}")
print(f"Params: {study.best_trial.params}")
