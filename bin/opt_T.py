import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F
import sys
import re
from sklearn.metrics import f1_score, precision_score, recall_score
import optuna
task = int(sys.argv[1])


def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    df = pd.DataFrame(data.tolist() if data.dtype == 'O' and isinstance(data[0], dict) else data)
    return df

def npy_preprocessor(filename):
    df = read_data(filename)
    return df['index'].values, df['inchi'].values, df['xyz'].values, df['chiral_centers'].values, df['rotation'].values

def calculate_iqr_bounds(data, k=1.5):
    """Calculates the IQR bounds for outlier detection.

    Args:
        data: A 1D numpy array of numerical data.
        k: The IQR multiplier (default: 1.5).

    Returns:
        A tuple containing the lower and upper bounds.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return lower_bound, upper_bound

def filter_outliers_iqr(xyz_arrays, k=1.5):
    """
    Filters outliers from xyz_arrays based on IQR of distances to centroid.

    Parameters:
    - xyz_arrays: List of arrays with shape (N, 8) where columns 0,1,2 are used
    - k: IQR multiplier

    Returns:
    - filtered_xyz: List of xyz_arrays without presumed outliers
    - valid_indices: Indices of the retained (non-outlier) xyz_arrays
    """
    
    distances_to_centroid = []
    for xyz in xyz_arrays:
      centroid = np.mean(xyz[:, :3], axis=0)
      distances = np.linalg.norm(xyz[:, :3] - centroid, axis=1)
      distances_to_centroid.extend(list(distances))

    
    lower_bound, upper_bound = calculate_iqr_bounds(np.array(distances_to_centroid), k)

    valid_indices = []
    for i, xyz in enumerate(xyz_arrays):
        centroid = np.mean(xyz[:, :3], axis=0)
        distances = np.linalg.norm(xyz[:, :3] - centroid, axis=1)
        
        # If all distances within an xyz_array are within bounds, consider it valid.
        if all(lower_bound <= dist <= upper_bound for dist in distances):
            valid_indices.append(i)

    filtered_xyz = [xyz_arrays[i] for i in valid_indices]

    return filtered_xyz, valid_indices

def filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    
    filtered_xyz_arrays, valid_indices = filter_outliers_iqr(xyz_arrays)
    
    if task == 0:
        filtered_index_array = [index_array[i] for i in valid_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in valid_indices]
        filtered_rotation_array = [rotation_array[i] for i in valid_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array

    if task == 1:
        #return only chiral_length <2
        filtered_indices = [i for i in valid_indices if len(chiral_centers_array[i]) < 2]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array

    elif task == 2:
        #only return chiral legnth < 5
        filtered_indices = [i for i in valid_indices if len(chiral_centers_array[i]) < 5]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array
    elif task == 3: 
        # Step 1: Filter indices where the length of chiral_centers_array is exactly 1 and the first tuple contains 'R' or 'S'
        filtered_indices = [i for i in valid_indices if len(chiral_centers_array[i]) == 1 and ('R' == chiral_centers_array[i][0][1] or 'S' == chiral_centers_array[i][0][1])]
        # Step 2: Create filtered arrays for index_array, xyz_arrays, chiral_centers_array, and rotation_array
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        
        # Step 5: Filter the rotation_array accordingly
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array
    
    elif task == 4:
        # only return chiral_length == 1
        filtered_indices = [i for i in valid_indices if len(chiral_centers_array[i]) == 1]
        filtered_index_array = [index_array[i] for i in filtered_indices]
        filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
        filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
        filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
        return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array
    elif task == 5:
        filtered_indices = [i for i in valid_indices]
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

# 121416 item, each associated with a 27 row, 8 col matrix, apply global normalization to col 0,1,2 Rescaling data to a [0, 1]

def reflect_wrt_plane(xyz, plane_normal=[0, 0, 1]):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    d = np.dot(xyz, plane_normal)
    return xyz - 2 * np.outer(d, plane_normal)

def rotate_xyz(xyz, angles):
    theta_x, theta_y, theta_z = np.radians(angles)
    Rx = np.array([[1,0,0],
                   [0,np.cos(theta_x),-np.sin(theta_x)],
                   [0,np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[ np.cos(theta_y),0,np.sin(theta_y)],
                   [0,1,0],
                   [-np.sin(theta_y),0,np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z),-np.sin(theta_z),0],
                   [np.sin(theta_z), np.cos(theta_z),0],
                   [0,0,1]])
    R = Rz @ Ry @ Rx
    return np.dot(xyz, R.T)

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
    # xyz_arrays: list of arrays each with shape (27, 8)
    # Normalize only columns 0,1,2 globally
    all_xyz = np.concatenate([xyz[:, :3] for xyz in xyz_arrays], axis=0) 
    min_val = all_xyz.min()
    max_val = all_xyz.max()

    norm_xyz = []
    for xyz in xyz_arrays:
        xyz_copy = xyz.copy()
        xyz_copy[:, :3] = (xyz_copy[:, :3] - min_val) / (max_val - min_val)
        norm_xyz.append(xyz_copy)
    return min_val, max_val, norm_xyz

def apply_normalization(xyz_arrays, min_val, max_val):
    norm_xyz = []
    for xyz in xyz_arrays:
        xyz_copy = xyz.copy()
        xyz_copy[:, :3] = (xyz_copy[:, :3] - min_val) / (max_val - min_val)
        norm_xyz.append(xyz_copy)
    return norm_xyz

def augment_dataset(index_array, xyz_arrays, chiral_centers_array, rotation_array, label_array, task):
    aug_idx, aug_xyz, aug_chiral, aug_rot, aug_label = list(index_array), list(xyz_arrays), list(chiral_centers_array), list(rotation_array), list(label_array)
    for i in range(len(index_array)):
        if len(chiral_centers_array[i]) == 1:
            reflected_xyz = xyz_arrays[i].copy()
            reflected_xyz[:, :3] = reflect_wrt_plane(xyz_arrays[i][:, :3], [0,0,1])
            reflected_label = label_array[i]
            if task == 3: reflected_label = 1 - reflected_label
            elif task in [4,5]: reflected_label = -reflected_label
            aug_idx.append(index_array[i])
            aug_xyz.append(reflected_xyz)
            aug_chiral.append(chiral_centers_array[i])
            aug_rot.append(rotation_array[i])
            aug_label.append(reflected_label)
    return aug_idx, aug_xyz, aug_chiral, aug_rot, aug_label

# Define device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_size, output_size, dropout_rate=0.3):
        """
        input_size: number of features per token (columns)
        """
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_size))
        self.dropout_embedding = nn.Dropout((dropout_rate))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout_transformer = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Expecting x of shape (N, 1, 27, input_size) or (N, 27, input_size).
        We'll handle transposing if necessary.
        """
        # If x is (N, 1, 27, input_size), flatten out the channel dim => (N, 27, input_size)
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)  # (N, 27, input_size)

        # If the last dimension isn't what we expect, we transpose.
        if x.size(-1) != self.embedding.in_features:
            x = x.transpose(1, 2)

        seq_len = x.size(1)  # number of tokens
        pos_encoding = self.positional_encoding[:, :seq_len, :]
        x = self.embedding(x) + pos_encoding

        x = self.dropout_embedding(x)
        x = self.transformer_encoder(x)     # (N, seq_len, hidden_size)
        x = self.dropout_transformer(x)

        x = torch.mean(x, dim=1)            # average pooling across seq_len
        x = self.fc_out(x)                  # final layer
        return x




def weight_decay(cnn_model, l2_lambda, device):
    _reg = 0.0
    for param in cnn_model.parameters():
        _reg += torch.norm(param, 2)**2
    _reg = (l2_lambda / 2) * _reg
    return _reg

def evaluate_model(model, test_loader, criterion, task, test):
    model.eval()
    all_labels = []
    all_predictions = []
    running_loss = 0.0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            # Reshape to Nx1x27x8 for CNN
            data = data.unsqueeze(1)  # Add channel dimension

            outputs = model(data)

            if task == 2:
                loss = criterion(outputs, labels.long())
                predictions = torch.argmax(outputs, dim=1)
            else:
                # Binary classification
                outputs = torch.sigmoid(outputs)
                
                # Ensure the output and labels have the same shape
                outputs = outputs.view(-1)  # Flatten model output to match labels
                labels = labels.view(-1)   # Flatten labels for consistency
                
                loss = criterion(outputs, labels)
                predictions = (outputs > 0.5).float()

            running_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    average_type = 'macro' if task == 2 else 'binary'
    precision = precision_score(all_labels, all_predictions, average=average_type, zero_division=0)
    recall = recall_score(all_labels, all_predictions, average=average_type, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average=average_type, zero_division=0)
    if test:
        cm = confusion_matrix(all_labels, all_predictions)
        print("\nConfusion Matrix:")
        print(cm)
    return avg_loss, accuracy, precision, recall, f1



def train_model(
    train_data, 
    val_data, 
    test_data, 
    task, 
    num_epochs=50, 
    batch_size=8, 
    learning_rate=0.0001, 
    l2_lambda=1e-4,
    patience=5,             # Number of epochs to wait before early stopping
    min_delta=1e-4          # Minimum improvement in val_loss to be considered "progress"
):
    # Unpack data
    train_idx, train_xyz, train_chiral, train_rot = train_data
    val_idx, val_xyz, val_chiral, val_rot         = val_data
    test_idx, test_xyz, test_chiral, test_rot     = test_data

    # === Normalize data ===
    min_val, max_val, train_xyz = normalize_xyz_train(train_xyz)
    val_xyz  = apply_normalization(val_xyz,   min_val, max_val)
    test_xyz = apply_normalization(test_xyz,  min_val, max_val)

    train_labels_np = np.array(generate_label_array(train_idx, train_xyz, train_chiral, train_rot, task))
    val_labels_np   = np.array(generate_label_array(val_idx,   val_xyz,   val_chiral,   val_rot,   task))
    test_labels_np  = np.array(generate_label_array(test_idx,  test_xyz,  test_chiral,  test_rot,  task))

    # === Convert to tensors ===
    train_tensor = torch.tensor(np.array(train_xyz), dtype=torch.float32)
    val_tensor   = torch.tensor(np.array(val_xyz),   dtype=torch.float32)
    test_tensor  = torch.tensor(np.array(test_xyz),  dtype=torch.float32)

    train_labels = torch.tensor(train_labels_np, dtype=torch.float32 if task != 2 else torch.long)
    val_labels   = torch.tensor(val_labels_np,   dtype=torch.float32 if task != 2 else torch.long)
    test_labels  = torch.tensor(test_labels_np,  dtype=torch.float32 if task != 2 else torch.long)

    train_dataset = TensorDataset(train_tensor, train_labels)
    val_dataset   = TensorDataset(val_tensor,   val_labels)
    test_dataset  = TensorDataset(test_tensor,  test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    print("Data loaded")

    # === Initialize the Transformer model & hyperparams ===
    input_size  = 8
    hidden_size = 128   
    num_heads   = 8
    num_layers  = 4
    output_size = 5 if task == 2 else 1  # 5-class for task 2, binary otherwise

    model = TransformerClassifier(
        input_size=input_size,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_size=hidden_size,
        output_size=output_size
    ).to(device)

    if task == 2:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    print("Model initialized")
    print("Hyperparams: ", learning_rate, l2_lambda, hidden_size, num_heads, num_layers, "betas:", 0.9, 0.999)

    # === Early Stopping Setup ===
    best_val_loss      = float('inf')
    patience_counter   = 0

    # === Training Loop ===
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)  # (N,1,27,8)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            if task == 2:
                loss = criterion(outputs, labels)
            else:
                outputs = torch.sigmoid(outputs)
                outputs = outputs.view(-1)
                labels  = labels.view(-1)
                loss    = criterion(outputs, labels)
            
            # Add weight decay
            loss = loss + weight_decay(model, l2_lambda, device)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # === Evaluate on Validation Data ===
        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate_model(
            model, val_loader, criterion, task, test=False
        )
        print(f"   Training Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val F1: {val_f1:.4f}, "
              f"Val Acc: {val_acc:.2f}%")

        # === Check Early Stopping Condition ===
        if val_loss < (best_val_loss - min_delta):
            best_val_loss    = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered. "
                  f"No improvement in validation loss for {patience} epochs.")
            break

    # === Final Test Evaluation ===
    test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate_model(
        model, test_loader, criterion, task, test=True
    )
    print(f"Final Test Loss: {test_loss:.4f}, "
          f"Accuracy: {test_acc:.2f}%, "
          f"Precision: {test_prec:.4f}, "
          f"Recall: {test_recall:.4f}, "
          f"F1: {test_f1:.4f}")

    return model


# Optuna objective function
def objective(trial):
    # Hyperparameter suggestions
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    l2_lambda = trial.suggest_loguniform('l2_lambda', 1e-6, 1e-2)
    num_epochs = trial.suggest_int('num_epochs', 10, 100)
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    num_layers = trial.suggest_int('num_layers', 2, 8)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    # Load and preprocess data
    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor('qm9_filtered.npy')

    if task in [3, 4, 5]:
        # Print label distribution as a ratio
        print("Distribution of Labels:")
        print(pd.Series(generate_label_array(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)).value_counts(normalize=True))

    print("\nTASK:", task)
    print(device)

    # Filter and split data
    filtered_index, filtered_xyz, filtered_chiral, filtered_rotation = filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)
    train_data, val_data, test_data = split_data(filtered_index, filtered_xyz, filtered_chiral, filtered_rotation)

    # Define model
    output_size = 5 if task == 2 else 1  # Multiclass for task 2, binary otherwise
    model = TransformerClassifier(
        input_size=8,  # Features per token
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout_rate=dropout_rate
    ).to(device)

    # Train model
    model = train_model(
        train_data, val_data, test_data,
        task=task,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        l2_lambda=l2_lambda
    )

    # Evaluate on validation set
    val_idx, val_xyz, val_chiral, val_rot = val_data
    val_labels_np = np.array(generate_label(val_idx, val_xyz, val_chiral, val_rot, task=0))
    val_tensor = torch.tensor(np.array(val_xyz), dtype=torch.float32)
    val_labels = torch.tensor(val_labels_np, dtype=torch.float32)
    val_dataset = TensorDataset(val_tensor, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss() if task != 2 else nn.CrossEntropyLoss()
    _, val_accuracy, _, _, val_f1 = evaluate_model(model, val_loader, criterion, task=0, test=False)

    # Return negative validation accuracy to maximize
    return -val_accuracy


# Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best trial:")
print(f"Value (F1): {-study.best_trial.value}")
print(f"Params: {study.best_trial.params}")
