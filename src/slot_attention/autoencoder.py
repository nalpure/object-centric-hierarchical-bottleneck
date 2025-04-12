import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from src.slot_attention.slot_attention import SlotAttention


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

def spatial_broadcast(slots, resolution):
    """
    Broadcast slot features to a 2D grid and collapse slot dimension.
    Args:
        slots: Slot features. Shape: [batch_size, num_slots, slot_size].
        resolution: Tuple of integers specifying width and height of grid.
    Returns:
        grid: Broadcasted slot features. Shape: [batch_size, width, height, slot_size].
    """
    slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
    grid = slots.repeat((1, resolution[0], resolution[1], 1))
    return grid

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.register_buffer('grid', build_grid(resolution))

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

class Encoder(nn.Module):
    def __init__(self, hid_dim, resolution,num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, slots_dim, resolution, num_channels):
        super().__init__()
        self.resolution = resolution
        self.conv1 = nn.ConvTranspose2d(slots_dim, hid_dim, 5, padding=2)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, padding=2)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, padding=2)
        self.conv4 = nn.ConvTranspose2d(hid_dim, (num_channels + 1), 3, padding=1)
        self.decoder_pos = SoftPositionEmbed(slots_dim, resolution)

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = x.permute(0,2,3,1)
        return x

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_channels, num_slots, num_iterations, slots_dim, encdec_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_channels: Number of channels in input image.
        num_iterations: Number of iterations in Slot Attention.
        slots_dim: Dimensionality of slot features.
        encdec_dim: Dimensionality of encoder/decoder features.
        latent_dim: Dimensionality of latent space (if projection head is used).
        """
        super().__init__()
        self.slots_dim = slots_dim
        self.encdec_dim = encdec_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_channels = num_channels

        self.encoder_cnn = Encoder(self.encdec_dim, self.resolution, self.num_channels)
        self.decoder_cnn = Decoder(self.encdec_dim, self.slots_dim, self.resolution, self.num_channels)

        self.fc1 = nn.Linear(encdec_dim, encdec_dim)
        self.fc2 = nn.Linear(encdec_dim, encdec_dim)

        self.enc_norm = nn.LayerNorm([self.resolution[0] * self.resolution[1], self.encdec_dim])

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            slot_size=self.slots_dim,
            in_features=self.encdec_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            mlp_hidden_size = 128)

    def forward(self, image, slots_init=None):
        slots, attention_scores = self.encode(image, slots_init)
        active_slots, background_slot = separate_slots(slots, attention_scores)
        recon_combined, recons, masks = self.decode(slots)
        return recon_combined, recons, masks, active_slots, background_slot

    def encode(self, image, slots_init=None):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = self.enc_norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, encdec_dim].

        # Slot Attention module.
        # `slots` has shape: [batch_size, num_slots, slot_size].
        slots, attention_scores = self.slot_attention(x, slots_init)
        return slots, attention_scores
    
    def decode(self, slots):
        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size = slots.shape[0]

        # Broadcast slot features to a 2D grid and collapse slot dimension.
        grid = spatial_broadcast(slots, self.resolution)
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].

        x = self.decoder_cnn(grid)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension
        x = x.reshape(batch_size, -1, x.shape[1], x.shape[2], x.shape[3])
        # `x` has shape: [batch_size, num_slots, width, height, num_channels+1].

        recons, masks = x.split([self.num_channels,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)  # + 1e-8
        
        # Recombine image.
        recon_combined = torch.sum(recons * masks, dim=1)  
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, num_channels, width, height].

        masks = masks.squeeze()  # [batch_size, width, height]

        return recon_combined, recons, masks
    

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