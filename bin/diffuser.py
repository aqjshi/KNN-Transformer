import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import torch.nn.functional as F
import sys
import optuna

task = int(sys.argv[1])

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
    elif task == 1:
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) < 2]
    elif task == 2:
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) < 5]
    elif task == 3:
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) == 1 and ('R' == chiral_centers_array[i][0][1] or 'S' == chiral_centers_array[i][0][1])]
    elif task == 4:
        filtered_indices = [i for i in range(len(index_array)) if len(chiral_centers_array[i]) == 1]
    elif task == 5:
        filtered_indices = [i for i in range(len(index_array))]

    filtered_index_array = [int(index_array[i]) for i in filtered_indices] 
    filtered_xyz_arrays = [xyz_arrays[i] for i in filtered_indices]
    filtered_chiral_centers_array = [chiral_centers_array[i] for i in filtered_indices]
    filtered_rotation_array = [rotation_array[i] for i in filtered_indices]
    return filtered_index_array, filtered_xyz_arrays, filtered_chiral_centers_array, filtered_rotation_array

def generate_label(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    if task == 0 or task == 1:
        return [1 if len(chiral_centers) > 0 else 0 for chiral_centers in chiral_centers_array]
    elif task == 2:
        return [len(chiral_centers) for chiral_centers in chiral_centers_array]
    elif task == 3:
        return [
            1 if chiral_centers and len(chiral_centers[0]) > 1 and 'R' == chiral_centers[0][1] else 0
            for chiral_centers in chiral_centers_array
        ]
    elif task == 4 or task == 5:
        return [1 if posneg[0] > 0 else 0 for posneg in rotation_array]

def generate_label_array(index_array, xyz_arrays, chiral_centers_array, rotation_array, task):
    return generate_label(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)

def reflect_wrt_plane(xyz, plane_normal=[0, 0, 1]):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    d = np.dot(xyz, plane_normal)
    return xyz - 2 * np.outer(d, plane_normal)

def rotate_xyz(xyz, angles):
    theta_x, theta_y, theta_z = np.radians(angles)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # Apply rotation only to the spatial dimensions (first 3 columns)
    rotated_xyz = xyz.copy()
    rotated_xyz[:, :3] = np.dot(xyz[:, :3], R.T)
    return rotated_xyz

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

# Updated Augmentation Functions
def add_noise(xyz_arrays, noise_level=0.05):
    noisy_xyz = []
    for xyz in xyz_arrays:
        noise = np.random.normal(0, noise_level, xyz[:, :3].shape)
        xyz_copy = xyz.copy()
        xyz_copy[:, :3] += noise
        noisy_xyz.append(xyz_copy)
    return noisy_xyz

def rotate_molecules(xyz_arrays):
    rotated_xyz = []
    for xyz in xyz_arrays:
        angles = np.random.uniform(0, 360, 3)  # Random angles for rotation
        rotated_xyz.append(rotate_xyz(xyz.copy(), angles))
    return rotated_xyz

def augment_dataset(index_array, xyz_arrays, chiral_centers_array, rotation_array, label_array, task):
    aug_idx, aug_xyz, aug_chiral, aug_rot, aug_label = list(index_array), list(xyz_arrays), list(chiral_centers_array), list(rotation_array), list(label_array)

    # Add noise to xyz data
    noisy_xyz = add_noise(xyz_arrays)
    aug_idx.extend(index_array)
    aug_xyz.extend(noisy_xyz)
    aug_chiral.extend(chiral_centers_array)
    aug_rot.extend(rotation_array)
    aug_label.extend(label_array)

    # Rotate molecules
    rotated_xyz = rotate_molecules(xyz_arrays)
    aug_idx.extend(index_array)
    aug_xyz.extend(rotated_xyz)
    aug_chiral.extend(chiral_centers_array)
    aug_rot.extend(rotation_array)
    aug_label.extend(label_array)

    return aug_idx, aug_xyz, aug_chiral, aug_rot, aug_label

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
def sinusoidal_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=1)
    return emb

# 2. & 3. Reverse Process and U-Net
class MoleculeUNet(nn.Module):
    def __init__(self, xyz_channels, ohe_channels, timesteps, embedding_dim, hidden_dims, dropout_rate, num_classes):
        super(MoleculeUNet, self).__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Concatenate xyz and ohe data along the channel dimension
        in_channels = xyz_channels + ohe_channels 
        
        self.conv_in = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
       
        self.encoder = nn.ModuleList([
            self.down_block(hidden_dims[i], hidden_dims[i+1], embedding_dim, dropout_rate)
            for i in range(len(hidden_dims)-1)
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1)
        )

        self.decoder = nn.ModuleList([
            self.up_block(hidden_dims[i], hidden_dims[i-1], embedding_dim, dropout_rate)
            for i in range(len(hidden_dims)-1, 0, -1)
        ])

        self.conv_out = nn.Conv1d(hidden_dims[0], xyz_channels + ohe_channels, kernel_size=3, padding=1)
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # Updated classifier head
        self.classifier_head = nn.Sequential(
            nn.Linear(xyz_channels + ohe_channels, num_classes) if num_classes > 2 else nn.Linear(xyz_channels + ohe_channels, 1),
            nn.Sigmoid() if num_classes == 2 else nn.Identity()
        )

    def forward(self, x, timesteps):
        t = self.time_mlp(timesteps)
        x = torch.cat([x, t.unsqueeze(-1).expand(-1, -1, x.size(2))], dim=1)
        
        x = self.conv_in(x)

        hiddens = []
        for i, layer in enumerate(self.encoder):
            x = layer(x, t)
            if i < len(self.encoder) - 1:  # Don't add hiddens from the last down block before bottleneck
                hiddens.append(x)

        x = self.bottleneck(x)
        for i, layer in enumerate(self.decoder):
            x = torch.cat([x, hiddens[len(hiddens)-i-1]], dim=1)
            x = layer(x, t)

        x = self.conv_out(x)
        x_for_classification = self.pooling(x).squeeze(-1)  # Pooling for classification
        classification_output = self.classifier_head(x_for_classification)

        return x, classification_output

    def down_block(self, in_channels, out_channels, embedding_dim, dropout_rate):
        return nn.Sequential(
            nn.Conv1d(in_channels + embedding_dim, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            self.pool  # Moved pool layer here to downsample after convolutions
        )
    def up_block(self, in_channels, out_channels, embedding_dim, dropout_rate):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

def ddpm_schedules(beta1, beta2, T, device):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32, device=device) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T, device).items():
            self.register_buffer(k, v)

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            self.device
        )  # t ~ Uniform(1, n_T)
        _ts_embed = sinusoidal_embedding(_ts, 160)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None] * x
            + self.sqrtmab[_ts, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # return MSE between added noise, and our predicted noise
        pred, classification_output = self.nn_model(x_t, _ts_embed)
        return self.loss_mse(noise, pred), classification_output, _ts

    def sample(self, n_sample, size, device, guide_w=0.0):
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
        for i in range(self.n_T, 0, -1):
            print(f"sampling timestep {i:3d}", end="\r")
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1)

            x_i = x_i.double()
            t_is = t_is.double()
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split prediction into diffusion and classification components
            eps, classification_output = self.nn_model(x_i, t_is)

            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        return x_i.float(), classification_output

class MoleculeDataset(Dataset):
    def __init__(self, xyz_arrays, labels, indices):
        self.xyz_arrays = xyz_arrays
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.xyz_arrays)

    def __getitem__(self, idx):
        xyz = self.xyz_arrays[idx]
        label = self.labels[idx]
        index = self.indices[idx]
        # Use torch.from_numpy for efficiency:
        return torch.from_numpy(xyz).float(), torch.tensor(label, dtype=torch.long if task == 2 else torch.float32), torch.tensor(index, dtype=torch.long)
def evaluate_model(model, test_loader, criterion, task, test):
    model.eval()
    all_labels = []
    all_predictions = []
    running_loss = 0.0

    with torch.no_grad():
        for data, labels, _ in test_loader:
            data, labels = data.to(device), labels.to(device)

            # Process the 'xyz' data
            x = data.permute(0, 2, 1)  # Change from (N, 27, 8) to (N, 8, 27)
            
            outputs, classification_output = model.sample(data.shape[0], (data.shape[2], data.shape[1]), device)
            classification_output = classification_output.squeeze()
            
            if task == 2:
                loss = criterion(classification_output, labels.long())
                predictions = torch.argmax(classification_output, dim=1)
            else:
                # Binary classification
                loss = criterion(classification_output, labels)
                predictions = (classification_output > 0.5).float()
            
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

def weight_decay(model, l2_lambda, device):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.norm(param, p=2) ** 2
    return l2_lambda * reg_loss

def train_diffusion_model(train_data, val_data, test_data, task, trial=None, num_epochs=2, batch_size=8, learning_rate=0.00001, l2_lambda=1e-4, timesteps=20, dropout_rate=0.1):
    train_idx, train_xyz, train_chiral, train_rot = train_data
    val_idx, val_xyz, val_chiral, val_rot = val_data
    test_idx, test_xyz, test_chiral, test_rot = test_data
    
    # Augment training data
    train_idx, train_xyz, train_chiral, train_rot, train_labels_np = augment_dataset(
        train_idx, train_xyz, train_chiral, train_rot, 
        generate_label_array(train_idx, train_xyz, train_chiral, train_rot, task), task
    )

    min_val, max_val, train_xyz = normalize_xyz_train(train_xyz)
    val_xyz = apply_normalization(val_xyz, min_val, max_val)
    test_xyz = apply_normalization(test_xyz, min_val, max_val)
    
    train_labels_np = np.array(generate_label_array(train_idx, train_xyz, train_chiral, train_rot, task))
    val_labels_np = np.array(generate_label_array(val_idx, val_xyz, val_chiral, val_rot, task))
    test_labels_np = np.array(generate_label_array(test_idx, test_xyz, test_chiral, test_rot, task))

    train_dataset = MoleculeDataset(train_xyz, train_labels_np, train_idx)
    val_dataset = MoleculeDataset(val_xyz, val_labels_np, val_idx)
    test_dataset = MoleculeDataset(test_xyz, test_labels_np, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # output_size = 5 if task == 2 else 1
    unet = MoleculeUNet(
        xyz_channels=3,
        ohe_channels=5,
        timesteps=timesteps,
        embedding_dim=160,
        hidden_dims=[128, 256, 512],
        dropout_rate=dropout_rate,
        num_classes = 5 if task == 2 else 1
    )
    model = DDPM(
        nn_model=unet,
        betas=(1e-4, 0.02),
        n_T=timesteps,
        device=device,
        drop_prob=dropout_rate
    ).to(device)

    if task == 2:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Early stopping variables
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum improvement required to consider it as improvement
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels, _) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Process the 'xyz' data
            x = inputs.permute(0, 2, 1)  # Change from (N, 27, 8) to (N, 8, 27)

            optimizer.zero_grad()

            loss, classification_output, _ = model(x, c=labels) # classification_output here is logits
            classification_loss = criterion(classification_output.squeeze(), labels)
            total_loss = loss + classification_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
        
        if epoch % 5 == 0:
            val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, task, test=False)
            test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, task, test=True)
            print(f"{epoch} Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
            
            # Report intermediate results to Optuna
            if trial:
                trial.report(val_acc, epoch)

            # Handle pruning based on the intermediate value.
            if trial and trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        else:
            val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, task, test=False)

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.2f}%")

        # Check for early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
            break
    
        if task == 0 and val_acc < 79:
            print(f"Training stopped due to low accuracy: {val_acc:.2f}%")
            break
        elif task == 1 and val_acc < 66:
            print(f"Training stopped due to low accuracy: {val_acc:.2f}%")
            break
        elif task == 2 and val_acc < 36:
            print(f"Training stopped due to low accuracy: {val_acc:.2f}%")
            break
        elif task == 3 and val_acc < 45:
            print(f"Training stopped due to low accuracy: {val_acc:.2f}%")
            break
        elif task == 4 and val_acc < 45:
            print(f"Training stopped due to low accuracy: {val_acc:.2f}%")
            break               
        elif task == 5 and val_acc < 45:
            print(f"Training stopped due to low accuracy: {val_acc:.2f}%")
            break

    test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate_model(model, test_loader, criterion, task, test=True)
    print(f"Final Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    return model, val_acc

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    l2_lambda = trial.suggest_loguniform('l2_lambda', 1e-7, 1e-3)
    timesteps = trial.suggest_categorical('timesteps', [5, 10, 20])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    num_epochs = trial.suggest_int('num_epochs', 10, 50)
    hidden_dims = trial.suggest_categorical('hidden_dims', [[64, 128, 256], [128, 256, 512]])

    print(f"Trial Hyperparameters: learning_rate={learning_rate}, batch_size={batch_size}, l2_lambda={l2_lambda}, timesteps={timesteps}, dropout_rate={dropout_rate}, num_epochs={num_epochs}")

    index_array, inchi_array, xyz_arrays, chiral_centers_array, rotation_array = npy_preprocessor('qm9_filtered.npy')

    if task == 3 or task == 4 or task == 5:
        print("Distribution of Labels:")
        print(pd.Series(generate_label_array(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)).value_counts(normalize=True))

    print("\nTASK:", task)
    print(device)

    filtered_index, filtered_xyz, filtered_chiral, filtered_rotation = filter_data(index_array, xyz_arrays, chiral_centers_array, rotation_array, task)
    train_data, val_data, test_data = split_data(filtered_index, filtered_xyz, filtered_chiral, filtered_rotation)

    model, val_accuracy = train_diffusion_model(
        train_data, val_data, test_data, task=task, trial=trial,
        num_epochs=num_epochs, batch_size=batch_size,
        learning_rate=learning_rate, l2_lambda=l2_lambda,
        timesteps=timesteps, dropout_rate=dropout_rate
    )
    return val_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best trial:")
print(f"Value (Accuracy): {study.best_trial.value}")
print(f"Params: {study.best_trial.params}")
