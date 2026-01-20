import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


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


def identify_background(slots, attention_scores):
    """
    Separates slots into active and background components while preserving
    the order of active slots within each batch.

    Args:
        slots: torch.Tensor of shape [B, N, D]
        attention_scores: torch.Tensor of shape [B, N, H*W]

    Returns:
        active_slots:      [B, N-1, D]
        background_slot:   [B, 1, D]
        active_attn:       [B, N-1, H*W]
        background_attn:   [B, 1, H*W]
    """
    B, N, D = slots.shape
    _, _, HW = attention_scores.shape

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
    active_attn  = attention_scores[active_mask].view(B, N-1, HW)

    return active_slots, background_slot, active_attn, background_attn


def order_slots_temporal(slots, attention_scores, prev_slots=None, prev_attention=None):
    """
    Orders slots by matching them to previous slots using a weighted
    combination of slot-vector and attention-map similarity.

    Args:
        slots: [B, N, D]
        attention_scores: [B, N, H*W]
        prev_slots: [B, N, D]
        prev_attention: [B, N, H*W]
    Returns:
        ordered_slots: [B, N, D]
        ordered_attention: [B, N, H*W]
    """
    if prev_slots is None or prev_attention is None:
        slots_active, slot_bg, attn_active, attn_bg = identify_background(slots, attention_scores)
        slots = torch.cat([slot_bg, slots_active], dim=1)
        attn = torch.cat([attn_bg, attn_active], dim=1)
    else:
        slots_active, slot_bg, attn_active, attn_bg = identify_background(slots, attention_scores)
        slots_active_prev = prev_slots[:, 1:, :]
        attn_active_prev = prev_attention[:, 1:, :]
        slots_active, attn_active, _ = match_slots_temporal(
            slots_active_prev, slots_active, attn_active_prev, attn_active)
        slots = torch.cat([slot_bg, slots_active], dim=1)
        attn = torch.cat([attn_bg, attn_active], dim=1)
    
    return slots, attn