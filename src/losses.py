from torch.nn import functional as F
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np


def slot_slot_contrastive_loss(slots, temperature=0.075, batch_contrast=True, criterion=torch.nn.CrossEntropyLoss()) -> torch.Tensor:
    """
    Slot-slot contrastive loss as defined in
    'Temporally Consistent Object-Centric Learning by Contrasting Slots'.

    Args:
        slots: (B, T, S, D) — slots over time
        temperature: float — τ from the paper
        batch_contrast: bool — switch between intra-video and inter-video contrastive loss
        criterion: loss function, default is CrossEntropyLoss

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
    Encourages a homogeneous attention map for at least one slot, which is assumed to be the background slot.

    Args:
    attention: [B, S, HW] or [B, S, THW]
    """
    std = torch.std(attention, dim=-1)
    loss = std.min(dim=1)[0].mean()
    return loss


def disentanglement_loss(z_orig, z_pert, latent_idx, magnitude, disentangle_type="closest_magnitude"):
    """
    z_orig:     [B, O, EI]
    z_pert:     [B, O, EI]
    latent_idx: [B]       (int indices 0..EI-1)
    magnitude:  [B]       (scalar magnitude per batch item)
    disentangle_type: str  -- "closest_magnitude", "max_response", "averaged_matching"
    Returns scalar loss.
    """
    B, O, EI = z_orig.shape

    if disentangle_type == "closest_magnitude":
        latent_idx_exp = latent_idx.view(B, 1, 1).expand(B, O, 1)
        z_o = torch.gather(z_orig, dim=2, index=latent_idx_exp).squeeze(-1)     # [B, O]
        z_p = torch.gather(z_pert, dim=2, index=latent_idx_exp).squeeze(-1)     # [B, O]
        mag = magnitude.view(B, 1)
        sq_diff = (z_p - z_o - mag) ** 2
        min_sq_diff = sq_diff.min(dim=1).values
        return min_sq_diff.mean()

    elif disentangle_type == "max_response":
        latent_idx_exp = latent_idx.view(B, 1, 1).expand(B, O, 1)
        z_o = torch.gather(z_orig, dim=2, index=latent_idx_exp).squeeze(-1)     # [B, O]
        z_p = torch.gather(z_pert, dim=2, index=latent_idx_exp).squeeze(-1)     # [B, O]

        pert_z_idx = torch.abs(z_p - z_o).argmax(dim=1).view(B, 1)
        z_p_pert = torch.gather(z_p, dim=1, index=pert_z_idx).squeeze(-1)  # [B]
        z_o_pert = torch.gather(z_o, dim=1, index=pert_z_idx).squeeze(-1)  # [B]
        diff = (z_p_pert - z_o_pert - magnitude) ** 2
        return diff.mean()
    
    elif disentangle_type == "averaged_matching":
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
    
    else:
        raise ValueError(f"Unknown disentanglement_type: {disentangle_type}")
    

def _r2_score(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    """R^2 for 2D tensors (N, D). Returns Python float."""
    # Flatten to 2D (N, D)
    y = y.reshape(-1, y.shape[-1]).to(dtype=torch.float32)
    y_hat = y_hat.reshape(-1, y_hat.shape[-1]).to(dtype=torch.float32)
    ss_res = torch.sum((y - y_hat) ** 2)
    y_mean = torch.mean(y, dim=0, keepdim=True)
    ss_tot = torch.sum((y - y_mean) ** 2)
    if ss_tot.item() == 0.0:
        return 0.0
    return float((1.0 - (ss_res / ss_tot)).item())


def _fit_linear_regression(flat_X: torch.Tensor, flat_Y: torch.Tensor) -> torch.Tensor:
    """
    Closed-form least-squares solution (no intercept): W = pinv(X) @ Y
    flat_X: (N, P), flat_Y: (N, L) -> returns W (P, L)
    """
    X = flat_X.to(dtype=torch.float32)
    Y = flat_Y.to(dtype=torch.float32)
    # pseudo-inverse on current device
    W = torch.pinverse(X) @ Y
    return W


def linear_mcc_loss(truth: torch.Tensor, pred: torch.Tensor) -> float:
    """
    Args:
        truth: (data_size, num_slots, latent_size)  -- ground-truth latents
        pred:  (data_size, num_slots, pred_size)    -- predicted slot representations

    Returns:
        final_score (float): the final R^2 score after iterative rematching & refitting.
    """
    # shapes
    data_size, num_slots, latent_size = truth.shape
    _, _, pred_size = pred.shape

    # flatten for initial regression
    flat_truth = truth.reshape(-1, latent_size).to(dtype=torch.float32)  # (N*S, L)
    flat_pred = pred.reshape(-1, pred_size).to(dtype=torch.float32)      # (N*S, P)

    # initial fit
    W = _fit_linear_regression(flat_pred, flat_truth)  # (P, L)
    regressed = flat_pred @ W                          # (N*S, L)
    old_score = _r2_score(flat_truth, regressed)
    print(f"Initial score: {old_score}")

    # helper: per-sample rematch using Hungarian (operates on CPU numpy)
    def rematch_batch(predictions: torch.Tensor, regression: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        predictions: (data_size, S, P)
        regression:  (data_size, S, L)  -- regressed predictions in GT latent space
        target:      (data_size, S, L)  -- ground-truth latents
        returns: new_predictions with same shape as predictions where for each sample i:
                 new_predictions[i] = predictions[i][perm_i]
        """
        N, S, P = predictions.shape
        new_preds = torch.empty_like(predictions)

        # loop per sample: Hungarian on CPU numpy
        for i in range(N):
            # cost matrix: ||target[a] - regression[b]||^2 over latent dims
            tgt = target[i].detach().cpu().numpy()        # (S, L)
            reg = regression[i].detach().cpu().numpy()    # (S, L)
            diff = tgt[:, None, :] - reg[None, :, :]      # (S, S, L)
            cost = np.sum(diff * diff, axis=2)            # (S, S)
            row_ind, col_ind = linear_sum_assignment(cost)

            # build permutation perm such that perm[row] = col
            perm = np.empty(S, dtype=np.int64)
            perm[row_ind] = col_ind

            # apply perm to predictions
            new_preds[i] = predictions[i][perm]
        return new_preds

    # iterative rematch loop
    converged = False
    while not converged:
        # reshape regressed to (data_size, num_slots, latent_size)
        regressed_per_sample = regressed.reshape(data_size, num_slots, latent_size)

        # perform rematch and permute predictions accordingly
        pred = rematch_batch(pred, regressed_per_sample, truth)

        # re-flatten and refit linear regression
        flat_pred = pred.reshape(-1, pred_size)
        W = _fit_linear_regression(flat_pred, flat_truth)
        regressed = flat_pred @ W
        score = _r2_score(flat_truth, regressed)
        print(f"New score: {score}")

        if score <= old_score:
            converged = True
        else:
            old_score = score

    return old_score