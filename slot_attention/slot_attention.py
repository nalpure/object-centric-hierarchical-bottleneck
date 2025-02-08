import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from typing import Union, Tuple

from slot_attention.utils import sinkhorn, minimize_entropy_of_sinkhorn, assert_shape


class SlotAttentionOriginal(nn.Module):
    def __init__(self, num_slots, dim, encdec_dim, iters = 3, eps = 1e-8, hidden_dim = 128, init_slots=True):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.dim = dim

        if init_slots:
            self.init_slots = nn.Embedding(num_slots, dim)
            nn.init.xavier_uniform_(self.init_slots.weight)
        else:
            self.init_slots = None
            self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(encdec_dim, dim)
        self.to_v = nn.Linear(encdec_dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(encdec_dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None, return_attn=False):
        b = inputs.shape[0]
        n_s = num_slots if num_slots is not None else self.num_slots

        if self.init_slots is None:
            mu = self.slots_mu.expand(b, n_s, -1)
            sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
            slots = mu + sigma * torch.randn(mu.shape, device=inputs.device)
        else:
            slots = self.init_slots.weight.expand(b, -1, -1)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
        
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, self.dim),
                slots_prev.reshape(-1, self.dim)
            )

            slots = slots.reshape(b, -1, self.dim)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        
        if return_attn:
            return slots, attn - self.eps
        
        return slots


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, encdec_dim, iters = 3, eps = 1e-8, hidden_dim = 128, init_slots=True):
        super().__init__()

        self.in_features = encdec_dim
        self.num_iterations = iters
        self.num_slots = num_slots
        self.slot_size = dim  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = hidden_dim
        self.epsilon = eps

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.in_features, self.slot_size, bias=False)

        # OT approach
        self.mlp_weight_input = nn.Linear(self.in_features, 1)
        self.mlp_weight_slots = nn.Linear(self.slot_size, 1)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        # self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")))
        # self.slots_log_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")))

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
    
    def step(self, slots, n_s, k, v, a, batch_size, num_inputs):
        slots_prev = slots
        slots = self.norm_slots(slots)
        
        b = self.mlp_weight_slots(slots).squeeze(-1).softmax(-1) * n_s  # <- this is new

        # Attention.
        q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
        assert_shape(q.size(), (batch_size, n_s, self.slot_size))

        # attn_norm_factor = self.slot_size ** -0.5
        # attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
        # attn_logits = -attn_logits
        attn_logits = torch.cdist(k, q)  # <- this is new
        # k = k / k.norm(dim=2, keepdim=True)
        # q = q / q.norm(dim=2, keepdim=True)
        # attn_logits = 1.0 - 6* torch.matmul(k, q.transpose(2, 1))

        # attn = F.softmax(attn_logits, dim=-1)
        attn_logits, p, q = minimize_entropy_of_sinkhorn(attn_logits, a, b, mesh_lr=5)   # <- this is new
        attn, _, _ = sinkhorn(attn_logits, a, b, u=p, v=q)  # <- this is new

        # `attn` has shape: [batch_size, num_inputs, num_slots].
        assert_shape(attn.size(), (batch_size, num_inputs, n_s))

        # Weighted mean.
        # attn = attn + self.epsilon
        # attn = attn / torch.sum(attn, dim=1, keepdim=True)
        updates = torch.matmul(attn.transpose(1, 2), v)
        # `updates` has shape: [batch_size, num_slots, slot_size].
        assert_shape(updates.size(), (batch_size, n_s, self.slot_size))

        # Slot update.
        # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
        slots = self.gru(
            updates.view(batch_size * n_s, self.slot_size),
            slots_prev.view(batch_size * n_s, self.slot_size),
        )
        slots = slots.view(batch_size, n_s, self.slot_size)
        assert_shape(slots.size(), (batch_size, n_s, self.slot_size))
        slots = slots + self.mlp(self.norm_mlp(slots))
        assert_shape(slots.size(), (batch_size, n_s, self.slot_size))

        # return only input variables which changed
        return slots, attn

    def forward(self, inputs: torch.Tensor, num_slots=None, num_iterations=None, slots_init=None):
        # `inputs` has shape [batch_size, num_inputs, inputs_size]. example: [256, 4096, 64]
        batch_size, num_inputs, inputs_size = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        n_i = num_iterations if num_iterations is not None else self.num_iterations
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        if slots_init is not None:
            slots = slots_init.clone()
        else:
            slots_init = torch.randn((batch_size, n_s, self.slot_size))
            slots_init = slots_init.type_as(inputs)
            slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        a = self.mlp_weight_input(inputs).squeeze(-1).softmax(-1) * n_s  # <- this is new

        # Multiple rounds of attention.
        for _ in range(n_i-1):
            slots, attn = self.step(slots, n_s, k, v, a, batch_size, num_inputs)
        
        slots = slots.detach() # detach for approx implicit diff
        slots, attn = self.step(slots, n_s, k, v, a, batch_size, num_inputs)            

        slots_init = slots.clone()
        return slots, attn.permute(0,2,1), slots_init # the last argument is None



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
        self.grid = build_grid(resolution).to("cuda")

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

class Encoder(nn.Module):
    def __init__(self, hid_dim, resolution, num_frames, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels*num_frames, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution).to("cuda")

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
    def __init__(self, hid_dim, slots_dim, resolution, num_frames, num_channels):
        super().__init__()
        self.resolution = resolution
        self.conv1 = nn.ConvTranspose2d(slots_dim, hid_dim, 5, stride=1, padding=2)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=1, padding=2)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=1, padding=2)
        self.conv4 = nn.ConvTranspose2d(hid_dim, (num_channels + 1) * num_frames, 3, stride=1, padding=1)
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
    def __init__(self, resolution, num_frames, num_slots, num_channels, num_iterations, slots_dim, encdec_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_channels: Number of channels in input image.
        num_iterations: Number of iterations in Slot Attention.
        slots_dim: Dimensionality of slot features.
        encdec_dim: Dimensionality of encoder/decoder features.
        small_arch: Whether to use a small architecture.
        latent_dim: Dimensionality of latent space (if projection head is used).
        """
        super().__init__()
        self.num_frames = num_frames
        self.slots_dim = slots_dim
        self.encdec_dim = encdec_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_channels = num_channels

        self.encoder_cnn = Encoder(self.encdec_dim, self.resolution, self.num_frames, self.num_channels)
        self.decoder_cnn = Decoder(self.encdec_dim, self.slots_dim, self.resolution, self.num_frames, self.num_channels)

        self.fc1 = nn.Linear(encdec_dim, encdec_dim)
        self.fc2 = nn.Linear(encdec_dim, encdec_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=self.slots_dim,
            encdec_dim=self.encdec_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128)

    def forward(self, image):
        return self.decode(self.encode(image))

    def encode(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:]).to(image.device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        # `slots` has shape: [batch_size, num_slots, slot_size].
        slots, attention_scores, slots_init = self.slot_attention(x)
        return slots
    
    def decode(self, slots):
        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size = slots.shape[0]
        width, height = self.resolution

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, width, height, 1))
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].

        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+num_frames].

        # Undo combination of slot and batch dimension
        x = x.reshape(batch_size, -1, x.shape[1], x.shape[2], x.shape[3])

        recons, masks = x.split([self.num_channels*self.num_frames,self.num_frames], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels*num_frames].
        # `masks` has shape: [batch_size, num_slots, width, height, num_frames].

        # Reconstruct channels
        recons = recons.view(batch_size, self.num_slots, width, height, self.num_frames, self.num_channels)
        recons = recons.permute(0,4,1,2,3,5)  # [batch_size, num_frames, num_slots, width, height, num_channels]
        masks = masks.unsqueeze(-1)
        masks = masks.permute(0,4,1,2,3,5)  # [batch_size, num_frames, num_slots, width, height, 1]

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=2)(masks)  # + 1e-8
        
        recon_combined = torch.sum(recons * masks, dim=2)  # Recombine image.
        recon_combined = recon_combined.permute(0,1,4,2,3)
        # `recon_combined` has shape: [batch_size, num_frames, 3, width, height].

        masks = masks.squeeze(-1)  # [batch_size, num_frames, width, height]
        if self.num_frames == 1:
            recon_combined = recon_combined.squeeze(1)
            masks = masks.squeeze(1)
        else:
            recon_combined = recon_combined.reshape(batch_size, -1, height, width)


        return recon_combined, recons, masks, slots

class ProjectionHead(nn.Module):
    """
    Projection head for projecting slot features to a single latent space property, as described in Mansouri et al. (2023).
    Tuned for slot dimensionality of 64.
    """
    def __init__(self, slots_dim):
        super().__init__()
        self.fc1 = nn.Linear(slots_dim, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, 16, bias=False)
        self.fc4 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = x.squeeze(-1) # shape: [batch_size, num_slots]
        return x

class DisentangledSlotAttentionAutoEncoder(SlotAttentionAutoEncoder):
    def __init__(self, resolution, num_slots, num_channels, num_iterations, slots_dim, encdec_dim, small_arch, latent_dim):
        super().__init__(resolution, num_slots, num_channels, num_iterations, slots_dim, encdec_dim, small_arch)
        self.latent_dim = latent_dim

        self.projection_heads = nn.ModuleList(ProjectionHead(slots_dim) for _ in range(latent_dim))

    def get_latents(self, slots):
        # apply projection heads
        batch_size = slots.shape[0]
        z = torch.empty(batch_size, self.num_slots, self.latent_dim, device=slots.device)  # shape: [batch_size, num_slots, latent_dim]
        for i, p in enumerate(self.projection_heads):
            z_i = p(slots) # shape: [batch_size, num_slots]
            z[:, :, i] = z_i
        return z