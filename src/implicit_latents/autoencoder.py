from torch import nn
import torch
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        #e.g. 76→64→32→16→5
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.BatchNorm1d(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

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
        self.ln1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

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
        self.ln1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.BatchNorm1d(hidden_dim)
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
        self.fc1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.ln1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.BatchNorm1d(hidden_dim)
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
            x: [B, T, O, E]
        Returns:
            z: [B, O, E + I]
        """
        B, T, O, E = x.shape
        I = self.implicit_dim
        H = self.hidden_dim
        TE = T * E

        # 1) collapse time+explicit into per‐object vector
        #    → source: [B, O, TE]
        source = x.permute(0, 2, 1, 3).reshape(B, O, TE)

        # 2) build all off‐diagonal indices (i != j)
        idx = torch.arange(O, device=x.device)
        ii = idx.repeat_interleave(O)   # [O*O] = [0,0,0,1,1,1,2,2,2]
        jj = idx.repeat(O)              # [O*O] = [0,1,2,0,1,2,0,1,2]
        mask = (ii != jj)
        ii = ii[mask]                   # [P]
        jj = jj[mask]                   # [P]
        P = ii.shape[0]                 # P = O*(O-1)

        # 3) gather neighbor pairs
        #    src_pairs, nbr_pairs: [B, P, TE]
        src_pairs = source[:, ii, :]    # object‐i as source
        nbr_pairs = source[:, jj, :]    # object‐j as neighbor

        # 4) flatten batch & pairs → [B*P, TE]
        flat_src = src_pairs.reshape(B * P, TE)
        flat_nbr = nbr_pairs.reshape(B * P, TE)

        # 5) single big edge‐encoding call (no self‐pairs)
        flat_edge = self.edge_encoder(flat_src, flat_nbr)  # [B*P, H]

        # 6) rebuild [B, P, H] then [B, O, O-1, H]
        edge_enc = flat_edge.view(B, P, H).view(B, O, O-1, H)

        # 7) mean over the O-1 neighbors → [B, O, H]
        edge_mean = edge_enc.mean(dim=2)

        # 8) prepare node inputs: flatten batch×object
        flat_source = source.reshape(B * O, TE)         # [B*O, TE]
        flat_edges  = edge_mean.reshape(B * O, H)       # [B*O, H]

        # 9) one‐shot node encoding → [B*O, E+I]
        flat_z = self.node_encoder(flat_source, flat_edges)

        # 10) reshape back → [B, O, E+I]
        return flat_z.view(B, O, E + I)



    def decode(self, z):
        """
        Args:
            z: [B, O, L]   (L = E+I)
        Returns:
            x: [B, T, O, E]
        """
        B, O, L = z.shape
        T, E, H = self.seq_len, self.explicit_dim, self.hidden_dim

        # 1) build off‐diagonal indices
        idx = torch.arange(O, device=z.device)
        ii = idx.repeat_interleave(O)
        jj = idx.repeat(O)
        mask = (ii != jj)
        ii = ii[mask]
        jj = jj[mask]
        P = ii.shape[0]  # O*(O-1)

        # 2) gather latent pairs → [B, P, L]
        src_pairs = z[:, ii, :]
        nbr_pairs = z[:, jj, :]

        # 3) flatten → [B*P, L]
        flat_src = src_pairs.reshape(B * P, L)
        flat_nbr = nbr_pairs.reshape(B * P, L)

        # 4) single big edge‐decoding → [B*P, H]
        flat_edge = self.edge_decoder(flat_src, flat_nbr)

        # 5) regroup → [B, P, H] → [B, O, O-1, H]
        edge_enc = flat_edge.view(B, P, H).view(B, O, O-1, H)

        # 6) mean over neighbors → [B, O, H]
        edge_mean = edge_enc.mean(dim=2)

        # 7) prepare node inputs
        flat_z      = z.reshape(B * O, L)         # [B*O, L]
        flat_edges  = edge_mean.reshape(B * O, H) # [B*O, H]

        # 8) one‐shot node decoding → [B*O, T*E]
        flat_x = self.node_decoder(flat_z, flat_edges)

        # 9) reshape → [B, O, T, E] then permute to [B, T, O, E]
        return flat_x.view(B, O, T, E).permute(0, 2, 1, 3)
