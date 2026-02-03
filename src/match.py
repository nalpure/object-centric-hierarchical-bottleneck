import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torch


def find_gt_slot_alignment(masks_pred: torch.Tensor,
                           masks: torch.Tensor):
    """
    Args:
        masks_pred: (T, S, H, W) predicted masks
        masks:      (T, S, H, W) ground-truth masks

    Returns:
        perm: (S,) permutation aligning predicted slots to GT objects
    """
    T, S, H, W = masks_pred.shape

    flat_pred = masks_pred.reshape(T, S, -1)
    flat_gt = masks.reshape(T, S, -1).float()

    # cost[i, j] = sum_t ||gt_i - pred_j||^2
    cost = torch.sum(
        (flat_gt[:, :, None, :] - flat_pred[:, None, :, :]) ** 2,
        dim=(0, 3)
    )  # (S, S)

    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

    perm = torch.as_tensor(col_ind, device=masks_pred.device)

    return perm


def match_slots_temporal(prev_slots, curr_slots, prev_attn, curr_attn, w_slot=1.0, w_attn=0.0):
    """
    Matches active slots between two timesteps (prev -> curr) using a weighted
    combination of slot-vector and attention-map similarity.

    Args:
        prev_slots: [B, S, D]
        curr_slots: [B, S, D]
        prev_attn:  [B, S, HW]
        curr_attn:  [B, S, HW]
        w_slot: weight for slot-vector similarity
        w_attn: weight for attention similarity
    Returns:
        curr_slots_reordered: [B, S, D]
        curr_attn_reordered:  [B, S, HW]
        assignment: LongTensor [B, S] (curr index assigned to each prev slot)
    """
    device = curr_slots.device
    B, S, D = prev_slots.shape

    # Normalize across last dim (vector/cosine) and attention maps
    prev_slots_n = F.normalize(prev_slots, dim=-1)     # (B,S,D)
    curr_slots_n = F.normalize(curr_slots, dim=-1)     # (B,S,D)
    prev_attn_n  = F.normalize(prev_attn, dim=-1)      # (B,S,HW)
    curr_attn_n  = F.normalize(curr_attn, dim=-1)      # (B,S,HW)

    # sim_slot[b] -> (S, S) similarity between prev_slots[b] and curr_slots[b]
    sim_slot = torch.matmul(prev_slots_n, curr_slots_n.transpose(1, 2))   # (B, S, S)
    sim_attn = torch.matmul(prev_attn_n, curr_attn_n.transpose(1, 2))     # (B, S, S)
    sim = w_slot * sim_slot + w_attn * sim_attn                            # (B, S, S)

    reordered_slots = torch.zeros_like(curr_slots)
    reordered_attn  = torch.zeros_like(curr_attn)
    assignments = torch.zeros((B, S), dtype=torch.long, device=device)

    for b in range(B):
        with torch.no_grad():
            sim_b = sim[b].cpu().numpy()    # (S, S), move to cpu for SciPy
            # we maximize similarity -> minimize -sim
            row_ind, col_ind = linear_sum_assignment(-sim_b)
            perm = torch.tensor(col_ind, device=device, dtype=torch.long)

        reordered_slots[b] = curr_slots[b, perm]
        reordered_attn[b]  = curr_attn[b, perm]
        assignments[b]     = perm

    return reordered_slots, reordered_attn, assignments


def reorder_slots_background_first(slots, attention_scores):
    """
    Reorders slots such that the background slot (lowest spatial std in attention scores) 
    becomes index 0 for each batch item. Remaining slots keep their relative order.

    Args:
        slots: [B, N, D] or [B, T, N, D]
        attention_scores: [B, N, H*W] or [B, T, N, H*W]
    """

    if slots.dim() == 4:
        B, T, N, D = slots.shape
        _, _, _, HW = attention_scores.shape
        slots = slots.permute(0, 2, 1, 3).reshape(B, N, T * D)
        attention_scores = attention_scores.permute(0, 2, 1, 3).reshape(B, N, T * HW)
        background_idx = reorder_slots_background_first(slots, attention_scores)
        slots = slots.view(B, N, T, D).permute(0, 2, 1, 3)
        attention_scores = attention_scores.view(B, N, T, HW).permute(0, 2, 1, 3)
        return background_idx

    B, N, D = slots.shape

    # 1. Identify background slot per batch (lowest spatial std)
    background_idx = torch.argmin(torch.std(attention_scores, dim=-1), dim=-1)  # [B]

    # 2. Gather background slot and its attention
    batch_idx = torch.arange(B, device=slots.device)
    background_slot = slots[batch_idx, background_idx].unsqueeze(1)      # [B,1,D]
    background_attn = attention_scores[batch_idx, background_idx].unsqueeze(1)  # [B,1,HW]

    # 3. Build mask and select active slots/attentions (preserve order)
    all_idx = torch.arange(N, device=slots.device).unsqueeze(0)          # [1,N]
    active_mask = all_idx != background_idx.unsqueeze(1)                 # [B,N]

    active_slots = slots[active_mask].view(B, N-1, D)
    active_attn  = attention_scores[active_mask].view(B, N-1, -1)

    slots[:] = torch.cat([background_slot, active_slots], dim=1)
    attention_scores[:] = torch.cat([background_attn, active_attn], dim=1)

    return background_idx