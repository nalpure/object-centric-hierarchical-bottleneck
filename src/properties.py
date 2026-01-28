from typing import List
import torch


# Define the mappings between property names and numerical codes
# Default: attributes are not ordered by explicit / implicit distinction
# Sorted: first explicit attributes (pos, hue), then implicit attributes (vel)
implicit_properties = ['vel_x', 'vel_y']
explicit_properties = ['pos_x', 'pos_y', 'hue']

STRING_TO_CODE_DEFAULT = {
    "pos_x": 0,
    "pos_y": 1,
    "vel_x": 2,
    "vel_y": 3,
    "hue": 4,
}

STRING_TO_CODE_SORTED = {
    "pos_x": 0,
    "pos_y": 1,
    "hue": 2,
    "vel_x": 3,
    "vel_y": 4,
}

CODE_TO_STRING_DEFAULT = {v: k for k, v in STRING_TO_CODE_DEFAULT.items()}
CODE_TO_STRING_SORTED = {v: k for k, v in STRING_TO_CODE_SORTED.items()}


def reorder_perturbation_indices(indices, shift=0):
    """Reorder perturbation indices to match the new STRING_TO_CODE mapping."""
    reordered_indices = []
    for idx in indices:
        prop_name = CODE_TO_STRING_DEFAULT[idx.item()]
        new_idx = STRING_TO_CODE_SORTED[prop_name]
        reordered_indices.append(new_idx + shift)
    return torch.tensor(reordered_indices, device=indices.device)


def get_implicit_codes() -> List[int]:
    return [STRING_TO_CODE_DEFAULT[prop] for prop in implicit_properties]


def get_explicit_codes() -> List[int]:
    return [STRING_TO_CODE_DEFAULT[prop] for prop in explicit_properties]