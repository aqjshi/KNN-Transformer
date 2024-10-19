import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KNearestNeighborAttention(nn.Module):
    def __init__(self, d_model, k_neighbors):
        super(KNearestNeighborAttention, self).__init__()
        self.k_neighbors = k_neighbors
        self.scale = 1.0 / np.sqrt(d_model)
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: Tensor of shape (batch_size, num_atoms, 3) representing the 3D positions.
            features: Tensor of shape (batch_size, num_atoms, d_model) representing the features (e.g., embeddings).
        """
        batch_size, num_atoms, _ = xyz.shape

        # Step 1: Compute pairwise distances between atoms (Euclidean distance)
        # (batch_size, num_atoms, num_atoms)
        distances = torch.cdist(xyz, xyz, p=2)

        # Step 2: Find the indices of the k-nearest neighbors (excluding the atom itself)
        # Use torch.topk to find the smallest distances (k+1 to exclude the atom itself)
        _, knn_indices = torch.topk(distances, self.k_neighbors + 1, largest=False, dim=-1)
        knn_indices = knn_indices[..., 1:]  # Exclude self (distance 0)

        # Step 3: Gather the features of the k-nearest neighbors
        knn_features = torch.gather(
            features.unsqueeze(1).expand(-1, num_atoms, -1, -1),  # Shape (batch_size, num_atoms, num_atoms, d_model)
            2,
            knn_indices.unsqueeze(-1).expand(-1, -1, -1, features.size(-1))  # Shape (batch_size, num_atoms, k_neighbors, d_model)
        )  # Shape (batch_size, num_atoms, k_neighbors, d_model)

        # Step 4: Compute scaled dot-product attention between the atom and its k-nearest neighbors
        query = features.unsqueeze(2)  # Shape (batch_size, num_atoms, 1, d_model)
        key = knn_features  # Shape (batch_size, num_atoms, k_neighbors, d_model)
        value = knn_features  # Shape (batch_size, num_atoms, k_neighbors, d_model)

        # Dot product attention
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale  # Shape (batch_size, num_atoms, 1, k_neighbors)
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape (batch_size, num_atoms, 1, k_neighbors)
        
        # Weighted sum of the values
        attention_output = torch.matmul(attention_weights, value).squeeze(2)  # Shape (batch_size, num_atoms, d_model)

        return attention_output  # Return the attended features
    
# Transformer Model Definition
class TransformerModelWithKNN(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_classes=1, k_neighbors=5):
        super(TransformerModelWithKNN, self).__init__()
        
        self.k_neighbors = k_neighbors
        
        # Embedding for xyz coordinates (columns 0, 1, 2)
        self.xyz_embedding = nn.Linear(3, d_model)
        
        # Embedding for atom types (one-hot encoded: columns 3-7)
        self.ohe_embedding = nn.Linear(5, d_model)

        # K-Nearest Neighbor Attention Layer
        self.knn_attention = KNearestNeighborAttention(d_model, k_neighbors)

        # Transformer encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True  # Set batch_first=True here
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_classes),
            nn.Sigmoid()  # For binary classification
        )
    
    def forward(self, x):
        # Split into xyz and one-hot encoding
        xyz = x[:, :, :3]  # Shape (batch_size, 27, 3)
        ohe = x[:, :, 3:8]  # Shape (batch_size, 27, 5)
        
        # Embed the xyz and one-hot encoding
        xyz_embedded = self.xyz_embedding(xyz)  # Shape (batch_size, 27, d_model)
        ohe_embedded = self.ohe_embedding(ohe)  # Shape (batch_size, 27, d_model)

        # Combine embeddings
        combined = xyz_embedded + ohe_embedded  # Element-wise addition (broadcasting)

        # Apply K-Nearest Neighbor Attention
        knn_attended_features = self.knn_attention(xyz, combined)

        # Pass through the transformer encoder
        transformer_out = self.transformer(knn_attended_features)  # Shape (batch_size, 27, d_model)
        
        # Pooling (reduce 27 atom embeddings into a single vector per molecule)
        pooled_out = transformer_out.mean(dim=1)  # Shape (batch_size, d_model)
        
        # Final classification
        out = self.fc(pooled_out)  # Shape (batch_size, 1)
        return out