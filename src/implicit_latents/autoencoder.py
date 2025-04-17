from torch import nn
import torch
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        #e.g. 76→64→32→16→5
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln3 = nn.LayerNorm(hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, source_node, edge_mean):
        # source_node: [B, input_dim]
        # edge_mean: [B, hidden_dim]
        # Concatenate the source node and edge mean
        x = torch.cat((source_node, edge_mean), dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x


class NodeDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        # e.g.: 69→64→32→16→12
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln3 = nn.LayerNorm(hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, z, aggregated_edge_mean):
        # z: [B, input_dim]
        # aggregated_edge_mean: [B, hidden_dim]
        x = torch.cat((z, aggregated_edge_mean), dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        # x: [B, explicit_dim]
        return x
    

class EdgeEncoder(nn.Module):
    """
    Encoding of two time sequences of explicit latents to one edge encoding.
    """
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        #e.g. 24->64->64->64
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(node_dim * 2, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, source, neighbor):
        # source: [B, node_dim]
        # neighbor: [B, node_dim]
        # Concatenate the two time sequences
        x = torch.cat((source, neighbor), dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        # x: [B, hidden_dim]
        return x
    

class EdgeDecoder(nn.Module):
    """
    Decoding of two latents to one edge encoding.
    """
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        #e.g. 10->32->64->64
        self.fc1 = nn.Linear(latent_dim * 2, 32)
        self.ln1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, source_latent, neighbor_latent):
        # source_latent: [B, latent_dim]
        # neighbor_latent: [B, latent_dim]
        # Concatenate the two latents
        x = torch.cat((source_latent, neighbor_latent), dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        # x: [B, hidden_dim]
        return x


class ImplicitLatentAutoEncoder(nn.Module):
    """
    Implicit Latent Autoencoder for encoding and decoding sequences of explicit latents.
    """
    def __init__(self, explicit_dim, implicit_dim, seq_len, hidden_dim):
        super().__init__()
        self.explicit_dim = explicit_dim
        self.implicit_dim = implicit_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        latent_dim = explicit_dim + implicit_dim
        
        self.node_encoder = NodeEncoder(explicit_dim * seq_len + hidden_dim, latent_dim, hidden_dim)
        self.node_decoder = NodeDecoder(latent_dim + hidden_dim, seq_len * explicit_dim, hidden_dim)
        self.edge_encoder = EdgeEncoder(explicit_dim * seq_len, hidden_dim)
        self.edge_decoder = EdgeDecoder(latent_dim, hidden_dim)
        

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, num_objects, explicit_dim)
        Returns:
            z: Tensor of shape (batch_size, num_objects, explicit_dim + implicit_dim)
        """
        B, T, O, E = x.shape
        I = self.implicit_dim
        final_latents = torch.empty((B, O, E + I), device=x.device)

        for o in range(O):
            edge_encodings = torch.empty((B, O - 1, self.hidden_dim), device=x.device)
            source_node = x[:, :, o, :].reshape(B, T * E)
            # source_node has shape (batch_size, seq_len * explicit_dim)
            neighbor_count = 0

            for n in range(O):
                if o != n:
                    neighbor_node = x[:, :, n, :].reshape(B, T * E)
                    edge_encoding = self.edge_encoder(source_node, neighbor_node)
                    edge_encodings[:, neighbor_count, :] = edge_encoding
                    neighbor_count += 1
                
            # Aggregate edge encodings by taking the mean
            edge_mean = edge_encodings.mean(dim=1)
            # edge_mean has shape (batch_size, hidden_dim)
            
            # Pass the source node and aggregated edge encodings to the node encoder
            final_latents[:, o, :] = self.node_encoder(source_node, edge_mean)
        
        return final_latents

        
    def decode(self, z):
        """
        Args:
            z: Tensor of shape (batch_size, num_objects, explicit_dim + implicit_dim)
        Returns:
            x: Tensor of shape (batch_size, seq_len, num_objects, explicit_dim)
        """
        batch_size = z.shape[0]
        num_objects = z.shape[1]

        x = torch.empty((batch_size, self.seq_len, num_objects, self.explicit_dim), device=z.device)

        for o in range(num_objects):
            source_latent = z[:, o, :]
            # source_latent has shape (batch_size, explicit_dim + implicit_dim)
            neighbor_count = 0
            edge_encodings = torch.empty((batch_size, num_objects-1, self.hidden_dim), device=z.device)
            for n in range(num_objects):
                if o != n:
                    neighbor_latent = z[:, n, :]
                    edge_encoding = self.edge_decoder(source_latent, neighbor_latent)
                    edge_encodings[:, neighbor_count, :] = edge_encoding
                    neighbor_count += 1
            
            # Aggregate edge encodings by taking the mean
            edge_mean = edge_encodings.mean(dim=1)
            # edge_mean has shape (batch_size, hidden_dim)
            # Pass the source latent and aggregated edge encodings to the node decoder
            x_o = self.node_decoder(source_latent, edge_mean)
            x_o = x_o.reshape(batch_size, self.seq_len, self.explicit_dim)
            x[:, :, o, :] = x_o

        return x