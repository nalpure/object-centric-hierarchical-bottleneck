import torch
from torch import nn


class EdgeEncoder(nn.Module):
    """ Given two sequences of explicit latents, encode them into an edge representation. """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # e.g. 2*12->64->64
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, source, neighbor):
        x = torch.cat((source, neighbor), dim=-1)
        return self.net(x)
    

class NodeEncoder(nn.Module):
    """ Given a sequence of explicit latents and aggregated edge info, encode into implicit latent. """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # e.g. 12+64 -> 64 -> 32 -> 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, source_node, edges_info):
        x = torch.cat((source_node, edges_info), dim=-1)
        return self.net(x)


class LatentEdgeEncoder(nn.Module):
    """ Given two latents containing explicit+implicit info, encode into an edge representation. """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # e.g. 2*5->32->32
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, source, neighbor):
        x = torch.cat((source, neighbor), dim=-1)
        return self.net(x)
        

class LatentTransition(nn.Module):
    """ Given a latent containing explicit and implicit info, produce delta for prediction of next explicit latent. """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # e.g. 5+64 -> 64 -> 64 -> 32 -> 16 ->3
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, z, edges_info):
        x = torch.cat((z, edges_info), dim=-1)
        return self.net(x)


class RelationalLatentDynamics(nn.Module):
    """
    Model for predicting future explicit latents by building an internal implicit latent representation.
    """
    def __init__(self, explicit_dim, implicit_dim, seq_len, edge_dim=64, latent_edge_dim=64, attn_temperature=0.5):
        super().__init__()
        self.E = explicit_dim
        self.I = implicit_dim
        self.T = seq_len
        self.H = edge_dim
        self.H_latent = latent_edge_dim
        self.attn_temperature = attn_temperature

        self.EI = self.E + self.I
        self.TE = self.T * self.E

        self.edge_encoder = EdgeEncoder(self.TE * 2, self.H)
        self.node_encoder_first = NodeEncoder(self.TE + self.H, self.I)
        self.node_encoder_current = NodeEncoder(self.TE + self.H, self.I)

        self.latent_edge_encoder = LatentEdgeEncoder(self.EI * 2, self.H_latent)
        self.latent_transition = LatentTransition(self.EI + self.H_latent, self.EI)

        
        # gate projection
        self.gate_proj = nn.Linear(self.H + self.EI, 1)


    def forward(self, z_explicit_seq, t_future, reconstruct=False):
        """
        Predict future explicit latents given a sequence of past explicit latents.
        Args:
            z_explicit_seq: torch.Tensor of shape [B, T, O, E]
                The sequence of explicit latents
            t_future: int
                Number of future time steps to predict
            reconstruct: bool
                Whether to reconstruct the input sequence
        Returns:
            seq_return: torch.Tensor of shape [B, T_out, O, E]
                The predicted sequence of explicit latents, where T_out = T + t_future if reconstruct else t_future
            z_first: torch.Tensor of shape [B, O, E + I] or None
                The latent at the first time step used for reconstruction, or None if reconstruct is False
        """
        B, T, O, E = z_explicit_seq.shape
        assert T == self.T
        assert E == self.E

        T_out = T * reconstruct + t_future

        if T_out <= 0:
            raise ValueError("Either reconstruct must be True or t_future > 0")
        
        z_explicit_diffs = z_explicit_seq[:, 1:, :, :] - z_explicit_seq[:, :-1, :, :]  # [B, T-1, O, E]
        z_explicit_first = z_explicit_seq[:, 0:1, :, :]  # [B, 1, O, E]
        z_explicit_current = z_explicit_seq[:, -1:, :, :]  # [B, 1, O, E]
        
        # Compute edge aggregations from first explicit latent and following diffs between
        source = torch.cat([z_explicit_first, z_explicit_diffs], dim=1)  # [B, T, O, E]
        source = source.permute(0, 2, 1, 3).reshape(B, O, -1)
        edge_agg = self.get_edges(source, self.edge_encoder)  # [B, O, H]

        seq_return = torch.empty(B, T_out, O, E, device=z_explicit_seq.device)
        z_first = None

        if reconstruct:
            # Build latent for first time step, rollout to reconstruct input sequence
            z_implicit_first = self.compute_implicit(source, edge_agg, self.node_encoder_first)  # [B, O, I]
            z_first = torch.cat([z_explicit_first.squeeze(1), z_implicit_first], dim=-1)  # [B, O, E + I]
            z_pred = self.rollout(z_first, num_steps=T)  # [B, T, O, E]
            seq_return[:, :T, :, :] = z_pred
            z_first = z_first 

        if t_future > 0:
            # Build latent for current time step, rollout to predict future sequence
            z_implicit_current = self.compute_implicit(source, edge_agg, self.node_encoder_current)  # [B, O, I]
            z_current = torch.cat([z_explicit_current.squeeze(1), z_implicit_current], dim=-1)  # [B, O, E + I]
            z_pred_future = self.rollout(z_current, num_steps=t_future)  # [B, t_future, O, E]
            seq_return[:, T * reconstruct:, :, :] = z_pred_future

        return seq_return, z_first
    

    def get_edges(self, source, edge_encoder: EdgeEncoder):
        """
        Antisymmetric, direction-preserving pairwise interactions.

        Args:
            source: [B, O, D]
        Returns:
            edge_agg: [B, O, H]
        """
        B, O, D = source.shape
        H = self.H
        device = source.device

        # 1) build unordered pairs i < j
        ii, jj = torch.triu_indices(O, O, offset=1, device=device)
        P = ii.shape[0]  # O*(O-1)/2

        # 2) gather unordered object pairs
        src_pairs = source[:, ii, :]   # [B, P, D]
        nbr_pairs = source[:, jj, :]   # [B, P, D]

        # 3) flatten for encoding
        flat_src = src_pairs.reshape(B * P, D)
        flat_nbr = nbr_pairs.reshape(B * P, D)

        # 4) encode a single directed interaction per unordered pair
        # IMPORTANT: direction is learned by encoder via (src, nbr)
        flat_edge = edge_encoder(flat_src, flat_nbr)  # [B*P, H]
        edge_pairs = flat_edge.view(B, P, H)           # [B, P, H]

        # 5) zero initialize per-object accumulator
        edge_agg = torch.zeros(B, O, H, device=device)

        # 6) scatter +f to i, -f to j  (action = -reaction)
        edge_agg.index_add_(1, ii, edge_pairs)
        edge_agg.index_add_(1, jj, -edge_pairs)

        return edge_agg
    

    def compute_implicit(self, source, edge_agg, node_encoder:NodeEncoder):
        """
        Compute implicit latent from source information and aggregated edges.
        Args:
            source: torch.Tensor of shape [B, O, TE]
                The sequence of explicit latents
            edge_agg: torch.Tensor of shape [B, O, H]
                The aggregated edge information per object
            node_encoder: NodeEncoder
                The node encoder to use
        """
        B, O, TE = source.shape
        I = self.I
        H = self.H

        # 1) prepare node inputs: flatten batch×object
        flat_source = source.reshape(B * O, TE)         # [B*O, TE]
        flat_edges  = edge_agg.reshape(B * O, H)       # [B*O, H]

        # 2) one‐shot node encoding → [B*O, I]
        flat_z_impl = node_encoder(flat_source, flat_edges)
        z_impl = flat_z_impl.view(B, O, I)

        return z_impl
    
    
    def predict(self, z):
        """
        Args:
            z: torch.Tensor of shape [B, O, E + I]
                The latent representation containing implicit information
        Returns:
            z_explicit_pred: torch.Tensor of shape [B, O, E + I]
                The predicted latent representation for the next time step
        """
        B, O, EI = z.shape
        H_latent = self.H_latent
        flat_z = z.reshape(B * O, EI)

        # 1) compute aggregated edges in latent space
        edge_agg = self.get_edges(z, self.latent_edge_encoder)  # [B, O, H_latent]
        flat_edges = edge_agg.reshape(B * O, H_latent)

        #2) compute new latents
        delta_z = self.latent_transition(flat_z, flat_edges).reshape(B, O, EI)  # [B, O, E + I]
        z_new = z + delta_z  # [B, O, E + I]
        return z_new
    

    def rollout(self, z, num_steps):
        """
        Args:
            z: torch.Tensor of shape [B, O, E + I]
                The latent representation containing implicit information
            num_steps: int
                Number of time steps to predict
        Returns:
            z_explicit_pred: torch.Tensor of shape [B, num_steps, O, E]
                The sequence of predicted explicit latents
        """
        B, O, EI = z.shape
        E = self.E
        z_explicit_pred = torch.zeros(B, num_steps, O, E, device=z.device)
        z_current = z

        for t in range(num_steps):
            # z_current: [B, O, E + I]
            z_tplus1 = self.predict(z_current)                  # predict next timestep
            z_explicit_pred[:, t, :, :] = z_tplus1[:, :, :E]    # store explicit part of prediction
            z_current = z_tplus1                                # update current latent for next step

        return z_explicit_pred