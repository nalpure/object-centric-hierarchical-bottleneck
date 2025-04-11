from torch import nn
import torch.nn.functional as F

    
class ObjectEncoder(nn.Module):
    """
    Encoder for encoding of object slots to latent space properties.
    """
    def __init__(self, slots_dim, feature_dim):
        super().__init__()
        self.fc1 = nn.Linear(slots_dim, 64, bias=True)
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 32, bias=True)
        self.fc4 = nn.Linear(32, 16, bias=True)
        self.fc5 = nn.Linear(16, feature_dim, bias=True)

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
        x = F.relu(x)
        x = self.fc5(x)
        x = x.squeeze(-1) 
        return x
    

class ObjectDecoder(nn.Module):
    """
    Decoder for decoding of object slots from latent space properties.
    """
    def __init__(self, feature_dim, slots_dim):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 16, bias=True)
        self.fc2 = nn.Linear(16, 32, bias=True)
        self.fc3 = nn.Linear(32, 64, bias=True)
        self.fc4 = nn.Linear(64, 64, bias=True)
        self.fc5 = nn.Linear(64, slots_dim, bias=True)

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
        x = F.relu(x)
        x = self.fc5(x)
        return x


class LatentAutoEncoder(nn.Module):
    def __init__(self, latent_dim, slots_dim):  
        super().__init__()      
        self.latent_dim = latent_dim
        self.slots_dim = slots_dim
        self.object_encoder = ObjectEncoder(slots_dim, latent_dim)
        self.object_decoder = ObjectDecoder(latent_dim, slots_dim)

    def forward(self, slots):
        """
        Args:
            slots: torch.Tensor of shape [batch_size, num_slots, slot_dim]
                The active slots
        Returns:
            torch.Tensor of shape [batch_size, num_slots, latent_dim]:
                The reconstructed active slots after latent decoding
            torch.Tensor of shape [batch_size, num_slots, latent_dim]:
                The latents of the active slots
        """
        z = self.encode(slots)
        slots_reconstructed = self.decode(z)
        return slots_reconstructed, z
    
    def encode(self, slots):
        """
        Args:
            slots: torch.Tensor of shape [batch_size, num_slots, slots_dim]
                The active slots
        Returns:
            torch.Tensor of shape [batch_size, num_slots, latent_dim]:
                The latents of the active slots
        """
        batch_size, num_slots, slots_dim = slots.shape
        latent_dim = self.latent_dim

        slots_flat = slots.reshape(batch_size * num_slots, slots_dim)
        z_flat = self.object_encoder(slots_flat)
        z = z_flat.reshape(batch_size, num_slots, latent_dim)
        return z

    
    def decode(self, z):
        """
        Args:
            z: torch.Tensor of shape [batch_size, num_slots, latent_dim]
                The latents of the active slots
        Returns:
            torch.Tensor of shape [batch_size, num_slots, slots_dim]:
                The reconstructed active slots after latent decoding
        """
        batch_size, num_slots, latent_dim = z.shape
        slots_dim = self.slots_dim

        z_flat = z.view(batch_size * num_slots, latent_dim)
        slots_reconstructed_flat = self.object_decoder(z_flat)
        slots_reconstructed = slots_reconstructed_flat.view(batch_size, num_slots, slots_dim)
        return slots_reconstructed



