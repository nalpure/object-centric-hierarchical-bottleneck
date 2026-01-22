import torch
from scipy.optimize import linear_sum_assignment


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


import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple


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


def get_linear_mcc(truth: torch.Tensor, pred: torch.Tensor) -> float:
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