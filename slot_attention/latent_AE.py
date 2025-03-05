import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from slot_attention.AE import SlotAttentionAutoEncoder

    
class ObjectEncoder(nn.Module):
    """
    Encoder for encoding of object slots to latent space properties.
    """
    def __init__(self, slots_dim, feature_dim):
        super().__init__()
        self.fc1 = nn.Linear(slots_dim, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 64, bias=True)
        self.fc4 = nn.Linear(64, feature_dim, bias=True)

    def forward(self, x):
        """
        Args: x: torch.Tensor of shape [batch_size, num_slots, slot_dim]
        Returns: x: torch.Tensor of shape [batch_size, num_slots, feature_dim]
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = x.squeeze(-1) 
        return x
    

class ObjectDecoder(nn.Module):
    """
    Decoder for decoding of object slots from latent space properties.
    """
    def __init__(self, feature_dim, slots_dim):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 64, bias=True)
        self.fc4 = nn.Linear(64, slots_dim, bias=True)

    def forward(self, x):
        """
        Args: x: torch.Tensor of shape [batch_size, num_slots, feature_dim]
        Returns: x: torch.Tensor of shape [batch_size, num_slots, slot_dim]
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


class LatentSlotAttentionAutoEncoder(SlotAttentionAutoEncoder):
    def __init__(self, resolution, num_frames, num_channels, num_slots, num_iterations, slots_dim, encdec_dim, latent_dim):
        super().__init__(resolution, num_frames, num_channels, num_slots, num_iterations, slots_dim, encdec_dim)
        
        self.explicit_latent_dim = 3

        self.latent_dim = latent_dim
        self.object_encoder = ObjectEncoder(slots_dim, self.explicit_latent_dim)
        self.object_decoder = ObjectDecoder(self.explicit_latent_dim, slots_dim)


    def forward(self, image, slots_init=None, obj_index=None, feat_index=None, magnitude=0):
        """
        Args:
            image: torch.Tensor of shape [batch_size, num_channels, height, width]
            slots_init: torch.Tensor of shape [batch_size, num_slots, slot_dim]
            obj_index: int For experiments: index of object to modify
            feat_index: int For experiments: index of object's latent feature to modify
            magnitude: float For experiments: magnitude of modification
        Returns:
            If training:
                slots_original_ordered: torch.Tensor of shape [batch_size, num_slots, slot_dim]
                slots_reconstructed_ordered: torch.Tensor of shape [batch_size, num_slots, slot_dim]
            If not training:
                recon_combined: torch.Tensor of shape [batch_size, num_channels, height, width]
                recons: torch.Tensor of shape [batch_size, num_channels, height, width]
                masks: torch.Tensor of shape [batch_size, 1, height, width]
                slots: torch.Tensor of shape [batch_size, num_slots, slot_dim]
                z: torch.Tensor of shape [batch_size, num_slots, latent_dim]        
        """
        
        slots, attention_scores = self.encode(image, slots_init=slots_init)
        # slots has shape: [batch_size, num_slots, slot_dim]
        # attention_scores has shape: [batch_size, num_slots, height * width]

        active_slots, background_slot = separate_slots(slots, attention_scores)
        
        z = self.object_encoder(active_slots)
        if obj_index is not None:
            print("z before", z)
            z[:, obj_index, feat_index] += magnitude
            print("z after", z)
        active_slots_reconstructed = self.object_decoder(z)

        slots_reconstructed_ordered = torch.cat((active_slots_reconstructed, background_slot), dim=1)

        if self.training:
            slots_original_ordered = torch.cat((active_slots, background_slot), dim=1)
            return slots_original_ordered, slots_reconstructed_ordered
        else:
            recon_combined, recons, masks, slots = self.decode(slots_reconstructed_ordered)
            return recon_combined, recons, masks, slots, z

def separate_slots(slots, attention_scores):
        """
        Separates slots into active slots and background slot based on standard deviation of attention scores.
        @param slots: torch.Tensor of shape [batch_size, num_slots, slot_dim]
        @param attention_scores: torch.Tensor of shape [batch_size, num_slots, height * width]
        @return: 
            active_slots: torch.Tensor of shape [batch_size, num_active_slots, slot_dim]
            background_slot: torch.Tensor of shape [batch_size, 1, slot_dim]
        """
        batch_size, n_slots, slot_dim = slots.shape

        # Find mask with least standard deviation
        background_slot_indices = torch.argmin(torch.std(attention_scores, dim=-1), dim=-1)
        background_slot_mask = torch.nn.functional.one_hot(background_slot_indices, num_classes=n_slots).bool()

        active_slots = slots[~background_slot_mask]
        background_slot = slots[background_slot_mask]
        
        active_slots = active_slots.view(batch_size, n_slots-1, slot_dim)
        background_slot = background_slot.view(batch_size, 1, slot_dim)

        return active_slots, background_slot
