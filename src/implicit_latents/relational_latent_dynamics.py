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
    

class ImplicitTransition(nn.Module):
    """ Given a latent containing aggregated edge info, produce delta for prediction of next implicit latent. """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # e.g. 5+32 -> 64 -> 32 -> 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

        # gate projection
        self.gate_proj = nn.Linear(input_dim, 1)

        # Initialize bias to favor "no update" (small gates at start)
        nn.init.constant_(self.gate_proj.bias, -2.0)  # gives sigmoid(-2) ≈ 0.12
        self.input_dim = input_dim

    def forward(self, source_node, edges_info):
        x = torch.cat((source_node, edges_info), dim=-1)
        delta = self.net(x) # [B*O, I]
        gate = torch.sigmoid(self.gate_proj(x)) # [B*O, 1]
        return delta * gate
    

class ExplicitTransition(nn.Module):
    """ Given a latent containing explicit and implicit info, produce delta for prediction of next explicit latent. """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # e.g. 5 -> 32 -> 64 -> 32 -> 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, implicit_latent):
        return self.net(implicit_latent)


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
        self.node_encoder_first = NodeEncoder((self.T * self.E) + self.H, self.I)
        self.node_encoder_current = NodeEncoder((self.T * self.E) + self.H, self.I)

        self.latent_edge_encoder = LatentEdgeEncoder(self.EI * 2, self.H_latent)
        self.implicit_transition = ImplicitTransition(self.EI + self.H_latent, self.I)
        self.explicit_transition = ExplicitTransition(self.I, self.E)


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
        
        source = z_explicit_seq.permute(0, 2, 1, 3).reshape(B, O, T*E)
        edge_agg = self.get_edges(source, self.edge_encoder)  # [B, O, H]
        seq_return = torch.empty(B, T_out, O, E, device=z_explicit_seq.device)
        z_first = None

        if reconstruct:
            # Build latent for first time step, rollout to reconstruct input sequence
            z_explicit_first = z_explicit_seq[:, 0, :, :]  # [B, O, E]
            z_implicit_first = self.compute_implicit(z_explicit_seq, edge_agg, self.node_encoder_first)  # [B, O, I]
            z_first = torch.cat([z_explicit_first, z_implicit_first], dim=-1)  # [B, O, E + I]
            z_pred = self.rollout(z_first, num_steps=T)  # [B, T, O, E]
            seq_return[:, :T, :, :] = z_pred
            z_first = z_first 

        if t_future > 0:
            # Build latent for current time step, rollout to predict future sequence
            z_explicit_current = z_explicit_seq[:, -1, :, :]  # [B, O, E]
            z_implicit_current = self.compute_implicit(z_explicit_seq, edge_agg, self.node_encoder_current)  # [B, O, I]
            z_current = torch.cat([z_explicit_current, z_implicit_current], dim=-1)  # [B, O, E + I]
            z_pred_future = self.rollout(z_current, num_steps=t_future)  # [B, t_future, O, E]
            seq_return[:, T * reconstruct:, :, :] = z_pred_future

        return seq_return, z_first


    def get_edges(self, source, edge_encoder:EdgeEncoder):
        """
        Args:
            z_explicit_seq: torch.Tensor of shape [B, O, D]
                The sequence of explicit latents
        Returns:
            torch.Tensor of shape [B, O, H]
                The aggregated edge representations per object
        """
        B, O, D = source.shape
        H = self.H

        # 2) build all off‐diagonal indices (i != j)
        idx = torch.arange(O, device=source.device)
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
        flat_src = src_pairs.reshape(B * P, D)
        flat_nbr = nbr_pairs.reshape(B * P, D)

        # 5) single big edge‐encoding call (no self‐pairs)
        flat_edge = edge_encoder(flat_src, flat_nbr)  # [B*P, H]

        # 6) rebuild [B, P, H] then [B, O, O-1, H]
        edge_enc = flat_edge.view(B, P, H).view(B, O, O-1, H)

        # 7) summation aggregation of the O-1 edges → [B, O, H]
        edge_agg = edge_enc.sum(dim=2)

        return edge_agg
    

    def compute_implicit(self, z_explicit_seq, edge_agg, node_encoder:NodeEncoder):
        B, T, O, E = z_explicit_seq.shape
        I = self.I
        H = self.H
        TE = T * E

        # 1) collapse time+explicit into per‐object vector
        #    → source: [B, O, TE]
        source = z_explicit_seq.permute(0, 2, 1, 3).reshape(B, O, TE)

        # 2) prepare node inputs: flatten batch×object
        flat_source = source.reshape(B * O, TE)         # [B*O, TE]
        flat_edges  = edge_agg.reshape(B * O, H)       # [B*O, H]

        # 3) one‐shot node encoding → [B*O, I]
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
        flat_z = z.reshape(B * O, EI)  # [B*O, EI]

        # 1) compute new explicit latent, given previous explicit latent + implicit latent
        flat_z_expl = flat_z[:, :self.E]  # [B*O, E]
        flat_z_impl = flat_z[:, self.E:]  # [B*O, I]
        flat_delta_z_expl = self.explicit_transition(flat_z_impl)  # [B*O, E]
        flat_z_expl_tplus1 = flat_z_expl + flat_delta_z_expl
        z_updated = torch.cat([flat_z_expl_tplus1, flat_z_impl], dim=-1).view(B, O, EI)
        
        # 2) compute aggregated edges in latent space
        edge_agg = self.get_edges(z_updated, self.latent_edge_encoder)  # [B, O, H_latent]
        
        # 3) one‐shot node transition
        flat_z_updated = z_updated.reshape(B * O, EI)
        flat_edges = edge_agg.reshape(B * O, H_latent) 
        flat_delta_z = self.implicit_transition(flat_z_updated, flat_edges)
        flat_z_impl_tplus1 = flat_z_impl + flat_delta_z # [B*O, I]

        # 4) concatenate explicit + implicit for next step
        flat_z_pred = torch.cat([flat_z_expl_tplus1, flat_z_impl_tplus1], dim=-1)  # [B*O, E + I]

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