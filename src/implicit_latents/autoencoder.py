from torch import nn
import torch
import torch.nn.functional as F

class EdgeEncoder(nn.Module):
    """
    Encoding of two time sequences of explicit latents to one edge encoding.
    """
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(node_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, source, neighbor):
        """
        Args:
            source: Tensor of shape (batch_size, node_dim)
            neighbor: Tensor of shape (batch_size, node_dim)
        Returns:
            x: Tensor of shape (batch_size, hidden_dim)
        """
        x = torch.cat((source, neighbor), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        return x
    

class EdgeDecoder(nn.Module):
    """
    Decoding of two latents to one edge encoding.
    """
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()

        if latent_dim > 16:
            raise ValueError("The sum of explicit_dim and implicit_dim should be less than or equal to 16.")  
        
        if hidden_dim < 64:
            raise ValueError("The hidden_dim should be greater than or equal to 64.")
              
        self.fc1 = nn.Linear(latent_dim * 2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, source_latent, neighbor_latent):
        """
        Args:
            source_latent: Tensor of shape (batch_size, latent_dim)
            neighbor_latent: Tensor of shape (batch_size, latent_dim)
        Returns:
            x: Tensor of shape (batch_size, hidden_dim)
        """
        x = torch.cat((source_latent, neighbor_latent), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        return x
    

class NodeEncoder(nn.Module):
    """
    Encoding of stacked explicit latents and an aggregated edge encoding to a single latent which includes implicit features.
    """
    def __init__(self, node_dim, edge_dim, output_dim):
        super().__init__()

        if output_dim > 16:
            raise ValueError("The sum of explicit_dim and implicit_dim should be less than or equal to 16.")

        self.fc1 = nn.Linear(node_dim + edge_dim, edge_dim)
        self.fc2 = nn.Linear(edge_dim, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, output_dim)

    
    def forward(self, source, aggregated_edges):
        """
        Args:
            source: Tensor of shape (batch_size, node_dim)
            aggregated_edges: Tensor of shape (batch_size, edge_dim)
        Returns:
            x: Tensor of shape (batch_size, output_dim)
        """
        x = torch.cat((source, aggregated_edges), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        return x
    

class NodeDecoder(nn.Module):
    """
    Decoding of a latent and an edge encoding to a sequence of explicit latents.
    """
    def __init__(self, explicit_dim, implicit_dim, edge_dim, seq_len):
        super().__init__()

        if explicit_dim * seq_len > 32:
            raise ValueError(f"The explicit_dim * seq_len should be less than or equal to 32 but got {explicit_dim * seq_len}.")
        if edge_dim < 64:
            raise ValueError("The edge_dim should be greater than or equal to 64.")
              
        self.seq_len = seq_len
        self.explicit_dim = explicit_dim
        self.fc1 = nn.Linear(explicit_dim + implicit_dim + edge_dim, edge_dim)
        self.fc2 = nn.Linear(edge_dim, edge_dim)
        self.fc3 = nn.Linear(edge_dim, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, explicit_dim * seq_len)

    def forward(self, latent, aggregated_edges):
        """
        Args:
            latent: Tensor of shape (batch_size, explicit_dim + implicit_dim)
            aggregated_edges: Tensor of shape (batch_size, edge_dim)
        Returns:
            x: Tensor of shape (batch_size, seq_len, explicit_dim)
        """
        x = torch.cat((latent, aggregated_edges), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.explicit_dim)
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
        self.edge_encoder = EdgeEncoder(explicit_dim * seq_len, hidden_dim)
        self.node_encoder = NodeEncoder(explicit_dim * seq_len, hidden_dim, latent_dim)
        self.edge_decoder = EdgeDecoder(latent_dim, hidden_dim)
        self.node_decoder = NodeDecoder(explicit_dim, implicit_dim, hidden_dim, seq_len)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, num_objects, explicit_dim)
        Returns:
            z: Tensor of shape (batch_size, num_objects, explicit_dim + implicit_dim)
        """
        batch_size = x.shape[0]
        num_objects = x.shape[2]
        latent_dim = self.explicit_dim + self.implicit_dim
        final_latents = torch.empty((batch_size, num_objects, latent_dim), device=x.device)

        for o in range(num_objects):
            edge_encodings = torch.empty((batch_size, num_objects-1, self.hidden_dim), device=x.device)
            source_node = x[:, :, o, :].reshape(batch_size, -1)
            # source_node has shape (batch_size, explicit_dim * seq_len)
            neighbor_count = 0

            for n in range(num_objects):
                if o != n:
                    neighbor_node = x[:, :, n, :].reshape(batch_size, -1)
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
            x[:, :, o, :] = self.node_decoder(source_latent, edge_mean)

        return x