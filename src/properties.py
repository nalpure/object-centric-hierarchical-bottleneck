from typing import List
import torch


# Define the mappings between property names and numerical codes
# Default: attributes are not ordered by explicit / implicit distinction
# Sorted: first explicit attributes (pos, hue), then implicit attributes (vel)
IMPLICIT_PROPERTIES = ['vel_x', 'vel_y']
EXPLICIT_PROPERTIES = ['pos_x', 'pos_y', 'hue']


DATASET_PROPERTY_CODES = {
    "pos_x": 0,
    "pos_y": 1,
    "vel_x": 2,
    "vel_y": 3,
    "hue": 4,
}

SORTED_PROPERTY_CODES = {
    "pos_x": 0,
    "pos_y": 1,
    "hue": 2,
    "vel_x": 3,
    "vel_y": 4,
}


def get_property_from_dataset_code(code: int) -> str:
    """Convert dataset property code to property name."""
    for prop, prop_code in DATASET_PROPERTY_CODES.items():
        if prop_code == code:
            return prop
    raise ValueError(f"Invalid property code: {code}")


def reorder_perturbation_indices(indices, shift=0):
    """Reorder perturbation indices to match the new STRING_TO_CODE mapping."""
    reordered_indices = []
    for idx in indices:
        prop_name = get_property_from_dataset_code(idx.item())
        new_idx = SORTED_PROPERTY_CODES[prop_name]
        reordered_indices.append(new_idx + shift)
    return torch.tensor(reordered_indices, device=indices.device)


def get_implicit_codes() -> List[int]:
    return [DATASET_PROPERTY_CODES[prop] for prop in IMPLICIT_PROPERTIES]


def get_explicit_codes() -> List[int]:
    return [DATASET_PROPERTY_CODES[prop] for prop in EXPLICIT_PROPERTIES]


def get_explicit_indices() -> List[int]:
    all_properties = IMPLICIT_PROPERTIES + EXPLICIT_PROPERTIES
    properties_sorted = sorted(all_properties, key=lambda x: DATASET_PROPERTY_CODES[x])
    return [properties_sorted.index(prop) for prop in EXPLICIT_PROPERTIES]