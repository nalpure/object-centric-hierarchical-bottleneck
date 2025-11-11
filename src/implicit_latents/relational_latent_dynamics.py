import torch
from torch import nn
import torch.nn.functional as F


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
    

class LatentNodeTransition(nn.Module):
    """ Given a latent containing explicit+implicit info and aggregated edge info, produce delta for prediction. """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # e.g. 5+32 -> 64 -> 32 -> 5
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
    

class RelationalLatentDynamics(nn.Module):
    """
    Model for predicting future explicit latents by building an internal implicit latent representation.
    """
    def __init__(self, explicit_dim, implicit_dim, seq_len, hidden_dim=64, hidden_dim_latent=64, attn_temperature=0.5):
        super().__init__()
        self.E = explicit_dim
        self.I = implicit_dim
        self.T = seq_len
        self.H = hidden_dim
        self.H_latent = hidden_dim_latent
        self.attn_temperature = attn_temperature

        self.EI = self.E + self.I
        self.TE = self.T * self.E

        self.edge_encoder = EdgeEncoder(self.TE * 2, self.H)
        self.agg_proj = nn.Linear(self.H, 1)
        self.node_encoder = NodeEncoder((self.T * self.E) + self.H, self.I)

        self.latent_edge_encoder = LatentEdgeEncoder(self.EI * 2, self.H_latent)
        self.latent_agg_proj = nn.Linear(self.H_latent, 1)
        self.latent_node_transition = LatentNodeTransition(self.EI + self.H_latent, self.EI)


    def forward(self, z_explicit_seq, t_future):
        """
        Predict future explicit latents given a sequence of past explicit latents.
        Args:
            z_explicit_seq: torch.Tensor of shape [B, T, O, E]
                The sequence of explicit latents
            t_future: int
                Number of future time steps to predict
        Returns:
            z_explicit_pred: torch.Tensor of shape [B, t_future, O, E]
                The latent representation containing implicit information
        """
        B, T, O, E = z_explicit_seq.shape
        assert T == self.T
        assert E == self.E

        z_impl = self.compute_implicit(z_explicit_seq)
        z_expl_t0 = z_explicit_seq[:, -1, :, :]  # [B, O, E]
        z_t0 = torch.cat((z_expl_t0, z_impl), dim=-1)  # [B, O, E + I]
        z_pred = self.rollout(z_t0, t_future)  # [B, t_future, O, E]

        return z_pred


    def compute_implicit(self, z_explicit_seq):
        """
        Args:
            z_explicit_seq: torch.Tensor of shape [B, T, O, E]
                The sequence of explicit latents
        Returns:
            torch.Tensor of shape [B, O, I]:
                A latent representation containing only implicit information
        """
        B, T, O, E = z_explicit_seq.shape
        I = self.I
        H = self.H
        TE = T * E

        # 1) collapse time+explicit into per‐object vector
        #    → source: [B, O, TE]
        source = z_explicit_seq.permute(0, 2, 1, 3).reshape(B, O, TE)

        # 2) build all off‐diagonal indices (i != j)
        idx = torch.arange(O, device=z_explicit_seq.device)
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

        # 7) attention-based aggregation of the O-1 edges → [B, O, H]
        attn_logits = self.agg_proj(edge_enc).squeeze(-1)
        attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True)[0]
        attn = torch.softmax(attn_logits / self.attn_temperature, dim=-1)
        edge_agg = (edge_enc * attn.unsqueeze(-1)).sum(dim=2)

        # 8) prepare node inputs: flatten batch×object
        flat_source = source.reshape(B * O, TE)         # [B*O, TE]
        flat_edges  = edge_agg.reshape(B * O, H)       # [B*O, H]

        # 9) one‐shot node encoding → [B*O, I]
        flat_z_impl = self.node_encoder(flat_source, flat_edges)
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
        
        # 1) build all off‐diagonal indices
        idx = torch.arange(O, device=z.device)
        ii = idx.repeat_interleave(O)
        jj = idx.repeat(O)
        mask = (ii != jj)
        ii = ii[mask]
        jj = jj[mask]
        P = ii.shape[0]  # O*(O-1)

        # 2) gather latent pairs → [B, P, EI]
        src_pairs = z[:, ii, :]
        nbr_pairs = z[:, jj, :]

        # 3) flatten → [B*P, EI]
        flat_src = src_pairs.reshape(B * P, EI)
        flat_nbr = nbr_pairs.reshape(B * P, EI)

        # 4) single big edge‐decoding → [B*P, H]
        flat_edge = self.latent_edge_encoder(flat_src, flat_nbr)

        # 5) regroup → [B, P, H] → [B, O, O-1, H]
        edge_enc = flat_edge.view(B, P, H_latent).view(B, O, O-1, H_latent)

        # 6) attention-based aggregation of the O-1 edges → [B, O, H]
        attn_logits = self.latent_agg_proj(edge_enc).squeeze(-1)
        attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True)[0]
        attn = torch.softmax(attn_logits / self.attn_temperature, dim=-1)
        edge_agg = (edge_enc * attn.unsqueeze(-1)).sum(dim=2)

        # 7) prepare node inputs: flatten batch×object
        flat_source = z.reshape(B * O, EI)               # [B*O, EI]
        flat_edges  = edge_agg.reshape(B * O, H_latent) # [B*O, H]

        #8) one‐shot node transition → [B*O, EI]
        flat_delta_z = self.latent_node_transition(flat_source, flat_edges)

        # 9) predict next latent
        flat_z_pred = flat_source + flat_delta_z

        return flat_z_pred.view(B, O, EI)
    

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