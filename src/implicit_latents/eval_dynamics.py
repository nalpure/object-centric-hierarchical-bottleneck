import numpy as np
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from os.path import exists

#from src.implicit_latents.autoencoder import ImplicitLatentAutoEncoder
from implicit_latents.relational_latent_dynamics import RelationalLatentDynamics
from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, identify_background, order_slots
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import IMG_CHANNELS, PerturbedImageSequenceDataset, create_trail, get_config_argument, load_config, plot_grid, plot_images, reorder_perturbation_indices, set_seed, DEVICE

NUM_OUTPUT_FIGS = 5
OUTPUT_DIR = "data/figures/"


def main():
    config_name = get_config_argument()
    config = load_config(config_name)
    config_SA = config["slot_attention"]
    config_explicit = config["explicit_latents"]
    
    set_seed(config_explicit["seed"])
    ckpt_path_SA = config_SA["ckpt_path"]
    ckpt_path_explicit = config_explicit["ckpt_path"]

    print(f"Loading observation test dataset: {config_SA['test_path']}")
    test_dataset = PerturbedImageSequenceDataset(config_SA["test_path"], config_SA["in_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=config_SA['batch_size'], shuffle=False, drop_last=True)
    
    T_past = 4 # TODO: config
    T_future = 4

    print("Loading model:", ckpt_path_SA)
    model_SA = SlotAttentionAutoEncoder(
        resolution=config_SA["resolution"],
        num_slots=config_SA["num_slots"],
        num_iterations=config_SA["num_iterations"], 
        num_channels=IMG_CHANNELS,
        slots_dim=config_SA["slots_dim"], 
        encdec_dim=config_SA["encdec_dim"]).to(DEVICE)
    model_SA.eval() 
    model_SA.load_state_dict(torch.load(ckpt_path_SA, weights_only=True)['model_state_dict'])

    print("Loading model:", ckpt_path_explicit)
    model_explicit = ExplicitLatentAutoEncoder(
        latent_dim=config_explicit["explicit_dim"], 
        slots_dim=config_SA["slots_dim"]
    ).to(DEVICE)
    model_explicit.eval()
    model_explicit.load_state_dict(torch.load(ckpt_path_explicit, weights_only=True)['model_AE_state_dict'])

    model_implicit = RelationalLatentDynamics(
        explicit_dim=config_explicit["explicit_dim"],
        implicit_dim=config_explicit["implicit_dim"],
        seq_len=T_past
    ).to(DEVICE)	
    model_implicit.eval()
    model_implicit.load_state_dict(torch.load(ckpt_path_explicit, weights_only=True)['model_implicit_state_dict'])

    normalize_slots = config_explicit["normalize"]

    if normalize_slots:
        norm_stats_path = config_explicit["train_path"].replace(".h5", "_norm_stats.pt")
        if exists(norm_stats_path):
            print("Loading normalization stats from", norm_stats_path)
            stats_slots = torch.load(norm_stats_path, weights_only=True)
            mean_slots = stats_slots["mean"].to(DEVICE)
            std_slots = stats_slots["std"].to(DEVICE)
            print(f"mean: {mean_slots}, std: {std_slots}")
        else:
            raise FileNotFoundError(f"Normalization stats file not found: {norm_stats_path}")
    else:
        mean_slots = 0.0
        std_slots = 1.0

    obs_loss = []
    slot_loss = []
    latent_loss = []
    criterion = torch.nn.MSELoss(reduction='mean')

    reconstruct = config_explicit["reconstruction_loss_weight"] > 0
    predict = config_explicit["prediction_loss_weight"] > 0
    T_future = T_future if predict else 0
    T_out = T_past * reconstruct + T_future

    for batch_idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            obs, pert, magnitude, obj_index, prop_index = (b.to(DEVICE) for b in batch)
            obs = obs[:, :T_past + T_future]
            pert = pert[:, :T_past + T_future]
            prop_index = reorder_perturbation_indices(prop_index)

            # ----- SLOT ATTENTION ENCODING / DECODING ------
            # obs -> slots
            active_slots, background_slots, active_slots_pert, background_slots_pert = encode_obs(obs, pert, model_SA)
            B, T, O, _ = active_slots.shape

            active_slots_norm = (active_slots - mean_slots) / std_slots
            active_slots_pert_norm = (active_slots_pert - mean_slots) / std_slots

            # slots -> obs
            obs_SA, _ = decode_slots(active_slots, background_slots, model_SA)

            # ----- EXPLICIT / IMPLICIT LATENT DYNAMICS ------
            # slots -> explicit
            z_explicit = encode_slots(active_slots_norm, model_explicit)
            H_latent = z_explicit.shape[3]

            # explicit -> slots -> obs
            slot_recon_norm = decode_explicit_latents(z_explicit, model_explicit)
            slot_recon = slot_recon_norm * std_slots + mean_slots
            obs_recon, _ = decode_slots(slot_recon, background_slots, model_SA)

            # explicit (past) -> implicit -> explicit (past and/or future)
            z_explicit_computed, z_orig = model_implicit(
                z_explicit[:, :T_past, :, :], 
                T_future, 
                reconstruct=reconstruct
            )

            if reconstruct:
                _, z_pert = model_implicit(
                    z_explicit[:, :T_past, :, :],
                    t_future=0,
                    reconstruct=reconstruct
                )

            # explicit (future) -> slot (future) -> obs (future)
            slot_computed_norm = decode_explicit_latents(z_explicit_computed, model_explicit)

            if batch_idx == 0:
                for i in range(min(1, B)):
                    print(f"Predicted explicit latents for sample {i}:")
                    print(z_explicit_computed[i].cpu().numpy().shape, z_explicit_computed[i].cpu().numpy())

            slot_computed = slot_computed_norm * std_slots + mean_slots 
            obs_computed, _ = decode_slots(
                slot_computed, 
                background_slots[:, T_past*reconstruct:T_past*reconstruct+T_future], 
                model_SA
            )

            first_compute_idx = T_past * (1 - reconstruct)
            obs_loss.append(criterion(obs_computed, obs[:, first_compute_idx:]).item())
            slot_loss.append(criterion(slot_computed_norm, active_slots_norm[:, first_compute_idx:]).item())
            latent_loss.append(criterion(z_explicit_computed, z_explicit[:, first_compute_idx:]).item())

            if batch_idx == 0 and reconstruct:
                for i in range(min(5, B)):
                    print(f"Perturbed ob {obj_index[i]}, index {prop_index[i]}, magnitude {magnitude[i].item():.3f}:")
                    print(z_orig[i].cpu().numpy(), " (original)")
                    print(z_pert[i].cpu().numpy(), " (perturbed)")
                    print((z_pert[i] - z_orig[i]).cpu().numpy(), " (difference)")
                    delta = torch.zeros_like(z_orig[i])
                    delta[obj_index[i], prop_index[i]] = magnitude[i]
                    print("loss: ", criterion(z_pert[i] - z_orig[i, :], delta).item())
                    print("\n-----\n")

        # ----- PLOTTING -----
        if batch_idx == 0:
            for i in range(min(NUM_OUTPUT_FIGS, obs.shape[0])):
                first_row = obs[i, first_compute_idx:].cpu()
                second_row = obs_SA[i, first_compute_idx:].cpu()
                third_row = obs_computed[i].cpu()
                rows = [first_row, second_row, third_row]

                save_path = OUTPUT_DIR + f"random_{i}.png"
                row_titles = ["Observation", "SA Reconstruction", "Predicted"]
                column_titles = [f"t={t}" for t in range((1-T_past)*reconstruct, T_future)]

                plot_grid(rows, row_titles, column_titles, save_path)

    obs_loss = np.array(obs_loss)
    print(f"Observation Loss: \t{obs_loss.mean():.6f} ± {obs_loss.std():.6f}")
    print(f"Slot Loss: \t{np.array(slot_loss).mean():.6f} ± {np.array(slot_loss).std():.6f}")
    print(f"Latent Loss: \t{np.array(latent_loss).mean():.6f} ± {np.array(latent_loss).std():.6f}")
    

def encode_obs(obs, pert, model_SA: SlotAttentionAutoEncoder):
    """
    Encodes the observation into slots.
    Args:
    - obs (torch.Tensor): Observation of shape [batch_size, seq_len, channels, height, width]
    - pert (torch.Tensor): Perturbation of shape [batch_size, seq_len, channels, height, width]
    - model_SA (SlotAttentionAutoEncoder): Slot Attention AutoEncoder model
    Returns:
    - active_slots_orig (torch.Tensor): Active slots of shape [batch_size, seq_len, num_objects, slots_dim]
    - background_slot_orig (torch.Tensor): Background slots of shape [batch_size, seq_len, 1, slots_dim]
    - active_slots_pert (torch.Tensor): Active slots of shape [batch_size, seq_len, num_objects, slots_dim]
    - background_slot_pert (torch.Tensor): Background slots of shape [batch_size, seq_len, 1, slots_dim]
    """
    B, T, C, H, W = obs.shape
    O = model_SA.num_slots - 1
    active_slots_orig = torch.empty((B, T, O, model_SA.slots_dim), device=DEVICE)
    background_slots_orig = torch.empty((B, T, 1, model_SA.slots_dim), device=DEVICE)
    active_slots_pert = torch.empty((B, T, O, model_SA.slots_dim), device=DEVICE)
    background_slots_pert = torch.empty((B, T, 1, model_SA.slots_dim), device=DEVICE)

    prev_slots_orig = None
    prev_attn_orig = None

    for t in range(T):
        # Encode / decode one frame
        slots_orig_t, attn_orig_t = model_SA.encode(obs[:, t], prev_slots_orig)
        slots_orig_t, attn_orig_t = order_slots(slots_orig_t, attn_orig_t, prev_slots_orig, prev_attn_orig)
        
        if t == 0:
            # At first time step, match slot ordering of perturbed to the one of original
            # For all other timesteps, use previous slots for matching
            prev_slots_pert = slots_orig_t
            prev_attn_pert = attn_orig_t

        slots_pert_t, attn_pert_t = model_SA.encode(pert[:, t], prev_slots_pert)
        slots_pert_t, attn_pert_t = order_slots(slots_pert_t, attn_pert_t, prev_slots_pert, prev_attn_pert)
        
        active_slots_orig[:, t] = slots_orig_t[:, 1:]
        background_slots_orig[:, t] = slots_orig_t[:, :1]

        active_slots_pert[:, t] = slots_pert_t[:, 1:]
        background_slots_pert[:, t] = slots_pert_t[:, :1]

        prev_slots_orig, prev_attn_orig = slots_orig_t, attn_orig_t
        prev_slots_pert, prev_attn_pert = slots_pert_t, attn_pert_t

    return active_slots_orig, background_slots_orig, active_slots_pert, background_slots_pert

def encode_slots(slots, model_explicit: ExplicitLatentAutoEncoder):
    """
    Encodes the slots into explicit latents.
    Args:
    - slots (torch.Tensor): Active slots of shape [batch_size, seq_len, num_objects, slots_dim]
    - model_explicit (ExplicitLatentAutoEncoder): Explicit Latent AutoEncoder model
    Returns:
    - z_explicit (torch.Tensor): Explicit latents of shape [batch_size, seq_len, num_objects, explicit_dim]
    """
    batch_size, seq_len, num_objects, slots_dim = slots.shape
    slots = slots.view(batch_size * seq_len * num_objects, slots_dim)
    z_explicit = model_explicit.encode(slots)
    z_explicit = z_explicit.view(batch_size, seq_len, num_objects, model_explicit.latent_dim)
    return z_explicit


def decode_slots(active_slots, background_slots, model_SA: SlotAttentionAutoEncoder):
    """
    Decodes the slots into images.
    Args:
    - slots (torch.Tensor): Active slots of shape [batch_size, seq_len, num_objects, slots_dim]
    - background_slots (torch.Tensor): Background slots of shape [batch_size, seq_len, 1, slots_dim]
    - model_SA (SlotAttentionAutoEncoder): Slot Attention AutoEncoder model
    Returns:
    - obs_recon (torch.Tensor): Reconstructed images of shape [batch_size, seq_len, channels, height, width]
    """
    batch_size, seq_len, num_objects, slots_dim = active_slots.shape

    slots = torch.concat((active_slots, background_slots), dim=2)
    slots = slots.view(batch_size * seq_len, num_objects + 1, slots_dim)
    obs_recon, _, masks = model_SA.decode(slots)
    obs_recon = obs_recon.view(batch_size, seq_len, IMG_CHANNELS, *model_SA.resolution)
    masks = masks.view(batch_size, seq_len, num_objects + 1, *model_SA.resolution)
    return obs_recon, masks


def decode_explicit_latents(z_explicit, model_explicit: ExplicitLatentAutoEncoder):
    """
    Decodes the explicit latents into slots.
    Args:
    - z_explicit (torch.Tensor): Explicit latents of shape [batch_size, seq_len, num_objects, explicit_dim]
    - model_explicit (ExplicitLatentAutoEncoder): Explicit Latent AutoEncoder model
    Returns:
    - slots_recon (torch.Tensor): Reconstructed slots of shape [batch_size, seq_len, num_objects, slots_dim]    
    """
    batch_size, seq_len, num_objects, explicit_dim = z_explicit.shape
    z_explicit = z_explicit.reshape(batch_size * seq_len, num_objects, explicit_dim)
    slots_recon = model_explicit.decode(z_explicit)
    slots_recon = slots_recon.view(batch_size, seq_len, num_objects, model_explicit.slots_dim)
    return slots_recon    


if __name__ == "__main__":
    main()