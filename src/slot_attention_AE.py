import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from slot_attention import SlotAttention


def spatial_broadcast(slots, resolution):
    """
    Broadcast slot features to a 2D grid and collapse slot dimension.
    Args:
        slots: Slot features. Shape: [batch_size, num_slots, slot_size].
        resolution: Tuple of integers specifying width and height of grid.
    Returns:
        grid: Broadcasted slot features. Shape: [batch_size, width, height, slot_size].
    """
    B, S, D = slots.shape
    slots = slots.reshape((B*S, D)).unsqueeze(1).unsqueeze(2) # [B*S, 1, 1, D]
    grid = slots.repeat((1, resolution[0], resolution[1], 1)) # [B*S, H, W, D]
    return grid

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

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
    def __init__(self, hid_dim, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, slots_dim, resolution, num_channels):
        super().__init__()
        self.resolution = resolution
        self.conv1 = nn.ConvTranspose2d(slots_dim, hid_dim, 5, stride=2, padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=2, padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=2, padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=2, padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=1, padding=2)
        self.conv6 = nn.ConvTranspose2d(hid_dim, (num_channels + 1), 3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
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

        self.decoder_initial_size = (resolution[0] // 16, resolution[1] // 16)

        self.encoder_cnn = Encoder(self.encdec_dim, self.num_channels)
        self.decoder_cnn = Decoder(self.encdec_dim, self.slots_dim, self.resolution, self.num_channels)

        self.fc1 = nn.Linear(encdec_dim, encdec_dim)
        self.fc2 = nn.Linear(encdec_dim, encdec_dim)

        self.enc_norm = nn.LayerNorm([self.resolution[0] * self.resolution[1], self.encdec_dim])

        self.encoder_pos = SoftPositionEmbed(self.encdec_dim, resolution)
        self.decoder_pos = SoftPositionEmbed(slots_dim, self.decoder_initial_size)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            slot_size=self.slots_dim,
            in_features=self.encdec_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            mlp_hidden_size = 128)

    def forward(self, image):
        slots, attn = self.encode(image)       
        recon_combined, recons, masks = self.decode(slots)
        return recon_combined, recons, masks, attn    

    def encode(self, image, slots_init=None):
        # `image` has shape: [batch_size, num_channels, width, height].
        B, C, W, H = image.shape
        assert W == self.resolution[0] and H == self.resolution[1], \
            f'Input image resolution ({W}x{H}) does not match model resolution ({self.resolution[0]}x{self.resolution[1]}).'
        assert C == self.num_channels, \
            f'Input image channels ({C}) does not match model channels ({self.num_channels}).'
        assert slots_init is None or slots_init.shape == (B, self.num_slots, self.slots_dim), \
            f'Initial slots shape {slots_init.shape} does not match required shape {(B, self.num_slots, self.slots_dim)}.'
        
        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = x.permute(0,2,3,1)
        # `x` has shape: [batch_size, width, height, encdec_dim].
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
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
        B, S, D = slots.shape
        assert S == self.num_slots and D == self.slots_dim, \
            f'Input slots shape {slots.shape} does not match required shape {(B, self.num_slots, self.slots_dim)}.'
        
        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size = slots.shape[0]

        # Broadcast slot features to a 2D grid and collapse slot dimension.
        x = spatial_broadcast(slots, self.decoder_initial_size)
        # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        # `x` has shape: [batch_size*num_slots, slot_size, width_init, height_init].
        x = self.decoder_cnn(x)
        x = x.permute(0,2,3,1)

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
        
        recon_combined = recon_combined.permute(0,3,1,2)    # [batch_size, num_channels, width, height].
        recons = recons.permute(0,1,4,2,3)                  # [batch_size, num_slots, num_channels, width, height].
        masks = masks.squeeze()                             # [batch_size, num_slots, width, height]
        
        return recon_combined, recons, masks
    

def order_slots(slots, attention_scores, prev_slots=None, prev_attention=None):
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
        slots_active, attn_active, _ = match_slots(
            slots_active_prev, slots_active, attn_active_prev, attn_active)
        slots = torch.cat([slot_bg, slots_active], dim=1)
        attn = torch.cat([attn_bg, attn_active], dim=1)
    
    return slots, attn


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


def match_slots(prev_slots, curr_slots, prev_attn, curr_attn, w_slot=1.0, w_attn=0.0):
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
