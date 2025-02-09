import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from slot_attention.AE import SlotAttentionAutoEncoder


class ProjectionHead(nn.Module):
    """
    Projection head for projecting slot features to a single latent space property, as described in Mansouri et al. (2023).
    Tuned for slot dimensionality of 64.
    """
    def __init__(self, slots_dim):
        super().__init__()
        self.fc1 = nn.Linear(slots_dim, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, 16, bias=False)
        self.fc4 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = x.squeeze(-1) # shape: [batch_size, num_slots]
        return x

class DisentangledSlotAttentionAutoEncoder(SlotAttentionAutoEncoder):
    def __init__(self, resolution, num_frames, num_channels, num_slots, num_iterations, slots_dim, encdec_dim, latent_dim):
        super().__init__(resolution, num_frames, num_channels, num_slots, num_iterations, slots_dim, encdec_dim)
        
        self.latent_dim = latent_dim
        self.projection_heads = nn.ModuleList(ProjectionHead(slots_dim) for _ in range(latent_dim))


    def forward(self, image, reconstruct=True):
        slots = self.encode(image)
        z = self.get_latents(slots)

        if reconstruct:
            return *self.decode(slots), z
        else:
            return z

    def get_latents(self, slots):
        # apply projection heads
        batch_size = slots.shape[0]
        z = torch.empty(batch_size, self.num_slots, self.latent_dim, device=slots.device)  # shape: [batch_size, num_slots, latent_dim]
        for i, p in enumerate(self.projection_heads):
            z_i = p(slots) # shape: [batch_size, num_slots]
            z[:, :, i] = z_i
        return z
    
def disentanglement_loss(z_obs, z_perturbed, magnitudes):
    """
    Returns the disentanglement loss for a batch of sample pairs.
    Uses matching to calculate: (z_perturbed - (z_obs + delta)).

    @param z_obs: torch.Tensor, [B, S, D]
        Latent representation of the original observation.
    @param z_perturbed: torch.Tensor, [B, S, D]
        Latent representation of the perturbed observation.
    @param magnitudes: torch.Tensor, [B]
        Magnitude of the perturbation.
    """
    latent_dim = z_obs.shape[2]

    eye = torch.eye(latent_dim, device=z_obs.device)                    # [D, D]
    deltas = eye.unsqueeze(0) * magnitudes[:, None, None]               # [B, D, D]

    z_obs_expanded = z_obs.unsqueeze(2).unsqueeze(2)                    # [B, S, 1, 1, D]
    z_perturbed_expanded = z_perturbed.unsqueeze(1).unsqueeze(3)        # [B, 1, S, 1, D]
    deltas_expanded = deltas.unsqueeze(1).unsqueeze(1)                  # [B, 1, 1, D, D]

    diff = z_perturbed_expanded - (z_obs_expanded + deltas_expanded)     # [B, S, S, D, D]
    diff_norm = torch.norm(diff, dim=-1)                                        # [B, S, S, D]
    losses = diff_norm.min(dim=-1).values.min(dim=-1).values.min(dim=-1).values # [B]

    return losses.sum()