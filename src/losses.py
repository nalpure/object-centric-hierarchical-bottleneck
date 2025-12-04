from torch.nn import functional as F
import torch


EPS = 1e-8

def slot_slot_contrastive_loss(slots, temperature=0.075, batch_contrast=True, criterion=torch.nn.CrossEntropyLoss()) -> torch.Tensor:
    """
    Inter-video slot-slot contrastive loss as defined in
    'Temporally Consistent Object-Centric Learning by Contrasting Slots'.
    Adds optional margin and symmetric variants.

    Args:
        slots: (B, T, S, D) — slots over time
        temperature: float — τ from the paper

    Returns:
        scalar loss
    """
    slots = F.normalize(slots, p=2.0, dim=-1)
    if batch_contrast:
        slots = slots.split(1)  # [1xTxKxD]
        slots = torch.cat(slots, dim=-2)  # 1xTxK*BxD
    s1 = slots[:, :-1, :, :]
    s2 = slots[:, 1:, :, :]
    ss = torch.matmul(s1, s2.transpose(-2, -1)) / temperature
    B, T, S, D = ss.shape
    ss = ss.reshape(B * T, S, S)
    target = torch.eye(S).expand(B * T, S, S).to(ss.device)
    loss = criterion(ss, target)
    return loss


def attention_loss(attention):
    """ 
    For each slot, compute the standard deviation of its attention values. This is assumed to be the background.
    The loss is the mean of the minimum standard deviations. 
    """
    B, S, N = attention.shape
    std = torch.std(attention, dim=-1)
    loss = std.min(dim=1)[0].mean()
    return loss


def disentanglement_loss(z_orig, z_pert, latent_idx, magnitude):
    """
    z_orig:     [B, O, EI]
    z_pert:     [B, O, EI]
    latent_idx: [B]       (int indices 0..EI-1)
    magnitude:  [B]       (scalar magnitude per batch item)
    Returns scalar loss.
    """
    B, O, EI = z_orig.shape
    assert z_orig.shape == z_pert.shape
    assert latent_idx.shape == (B,)
    assert magnitude.shape == (B,)

    device = z_orig.device
    delta = z_pert - z_orig                      # [B, O, EI]

    # Build mask of shape [O, O] where mask[c, j] = 1 if j == c else 0
    eye_O = torch.eye(O, device=device)         # [O, O]

    # We'll build target_exp shaped [B, O_choice, O_object, EI]
    # Start with zeros:
    target_exp = torch.zeros((B, O, O, EI), device=device)  # [B, O_choice, O_object, EI]

    # We need to set target_exp[b, c, j, latent_idx[b]] = magnitude[b] if j == c
    # Use broadcasting: eye_O[None, :, :] has shape [1, O, O]
    # Expand magnitude and latent indices to match batch/hypothesis dims
    mask = eye_O[None, :, :].expand(B, -1, -1)          # [B, O, O]
    # latent_idx expanded to [B, 1, 1] so it can index into the EI dim
    lat_idx_exp = latent_idx.view(B, 1, 1)              # [B, 1, 1]
    mag_exp     = magnitude.view(B, 1, 1)               # [B, 1, 1]

    # Use scatter to place magnitude at the correct latent index only where mask==1
    # For scatter we need indices of shape [B, O, O, 1] and values shape [B, O, O, 1]
    idx_for_scatter = lat_idx_exp.expand(B, O, O).unsqueeze(-1)  # [B, O, O, 1]
    values_for_scatter = (mask.unsqueeze(-1) * mag_exp.unsqueeze(-1))  # [B, O, O, 1]

    # scatter_ along last dim (dim=-1)
    target_exp.scatter_(-1, idx_for_scatter, values_for_scatter)  # places magnitude only where mask==1

    # Expand delta to compare: delta_exp [B, 1, O, EI] so it broadcasts with target_exp
    delta_exp = delta.unsqueeze(1)  # [B, 1, O, EI]

    # Compute squared error per latent -> mean over EI gives [B, O_choice, O_object]
    loss_matrix = (delta_exp - target_exp).pow(2).mean(dim=-1)  # [B, O, O]

    # For a hypothesis 'choice' we average the per-object errors across objects:
    hypothesis_loss = loss_matrix.mean(dim=-1)  # [B, O]

    # pick best hypothesis per batch item
    best_loss = hypothesis_loss.min(dim=1).values  # [B]

    return best_loss.mean()
