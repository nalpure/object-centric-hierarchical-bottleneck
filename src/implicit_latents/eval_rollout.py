import numpy as np
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from os.path import exists

#from src.implicit_latents.autoencoder import ImplicitLatentAutoEncoder
from implicit_latents.relational_latent_dynamics import RelationalLatentDynamics
from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, identify_background, order_slots
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import IMG_CHANNELS, PerturbedImageSequenceDataset, create_trail, get_config_argument, load_config, plot_grid, plot_images, reorder_perturbation_indices, save_gif_from_array, set_seed, DEVICE

NUM_OUTPUT_FIGS = 15
OUTPUT_DIR = "data/figures/"


def main():
    config_name = get_config_argument()
    config = load_config(config_name)
    config_SA = config["slot_attention"]
    config_explicit = config["explicit_latents"]
    config_implicit = config["implicit_latents"]
    
    set_seed(config_explicit["seed"])
    ckpt_path_SA = config_SA["ckpt_path"]
    ckpt_path_explicit = config_explicit["ckpt_path"]
    ckpt_path_implicit = config_implicit["ckpt_path"]

    print(f"Loading observation test dataset: {config_SA['test_path']}")
    test_dataset = PerturbedImageSequenceDataset(config_SA["test_path"], config_SA["in_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    
    T = next(iter(test_dataloader))[0].shape[1]
    T_past = 4 # TODO: config
    T_future = 120

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
    model_explicit.load_state_dict(torch.load(ckpt_path_explicit, weights_only=True)['model_state_dict'])

    model_implicit = RelationalLatentDynamics(
        explicit_dim=config_explicit["explicit_dim"],
        implicit_dim=config_implicit["latent_dim"] - config_explicit["explicit_dim"],
        seq_len=T_past,
        edge_dim=config_implicit["edge_dim"],
        latent_edge_dim=config_implicit["latent_edge_dim"]
    ).to(DEVICE)	
    model_implicit.eval()
    model_implicit.load_state_dict(torch.load(ckpt_path_implicit, weights_only=True)['model_state_dict'])

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

    obs_pred_loss = []
    slot_pred_loss = []
    slot_recon_loss = []
    criterion = torch.nn.MSELoss(reduction='mean')


    sample = next(iter(test_dataloader))
    obs, pert, magnitude, obj_index, prop_index = (s.to(DEVICE) for s in sample)
    prop_index = reorder_perturbation_indices(prop_index)

    with torch.no_grad():
        # ----- SLOT ATTENTION ENCODING / DECODING ------
        # obs -> slots
        active_slots, background_slots, active_attn, background_attn = encode_obs(obs, model_SA)
        active_slots_norm = (active_slots - mean_slots) / std_slots

        # slots -> obs
        obs_SA, _ = decode_slots(active_slots, background_slots, model_SA)

        # ----- EXPLICIT / IMPLICIT LATENT DYNAMICS ------
        # slots -> explicit
        z_explicit = encode_slots(active_slots_norm, model_explicit)
        z_explicit_past = z_explicit[:, :T_past]

        recursive = True
        
        if recursive:
            z_explicit_prediction = torch.empty((1, T_future, z_explicit.shape[2], z_explicit.shape[3]), device=DEVICE)
            predict_at_once = 4

            for i in range(T_future // predict_at_once):
                # explicit (past) -> implicit -> explicit (future)
                z_explicit_prediction_current, _ = model_implicit(z_explicit_past, predict_at_once)
                z_explicit_prediction[:, i*predict_at_once:(i+1)*predict_at_once] = z_explicit_prediction_current
                if predict_at_once < T_past:
                    """z_explicit_past = torch.roll(z_explicit_past, -predict_at_once, dims=1)
                    z_explicit_past[:, -predict_at_once:] = z_explicit_prediction_current"""
                    z_explicit_past = torch.cat((
                        z_explicit_past[:, predict_at_once:], 
                        z_explicit_prediction_current), dim=1)
                elif predict_at_once == T_past:
                    z_explicit_past = z_explicit_prediction_current
                else:
                    z_explicit_past = z_explicit_prediction_current[:, -T_past:]
        else:
            z_explicit_prediction, _ = model_implicit(z_explicit_past, T_future)

        # explicit (future) -> slot (future) -> obs (future)
        slot_prediction_norm = decode_explicit_latents(z_explicit_prediction, model_explicit)

        slot_prediction = slot_prediction_norm * std_slots + mean_slots 
        background_slots = background_slots[:, :1].repeat(1, T_past + T_future, 1, 1)
        obs_prediction, _ = decode_slots(slot_prediction, background_slots[:, T_past:], model_SA)


    # ----- CREATING VIDEO -----
    imgs = []
    for t in range(3, T_future):
        imgs_t = [
            obs_prediction[0, t-3].cpu().numpy(),
            obs_prediction[0, t-2].cpu().numpy(),
            obs_prediction[0, t-1].cpu().numpy(),
            obs_prediction[0, t].cpu().numpy()]
        
        trail_t = create_trail(np.array(imgs_t))
        imgs.append(trail_t)

    save_gif_from_array(
        np.array(imgs),
        OUTPUT_DIR + "rollout.gif"
    )
    

def encode_obs(obs, model_SA: SlotAttentionAutoEncoder, prev_slots=None, prev_attn=None):
    """
    Encodes the observation into slots.
    Args:
    - obs (torch.Tensor): Observation of shape [batch_size, seq_len, channels, height, width]
    - model_SA (SlotAttentionAutoEncoder): Slot Attention AutoEncoder model
    Returns:
    - active_slots (torch.Tensor): Active slots of shape [batch_size, seq_len, num_objects, slots_dim]
    - background_slot (torch.Tensor): Background slots of shape [batch_size, seq_len, 1, slots_dim]
    """
    B, T, C, H, W = obs.shape
    O = model_SA.num_slots - 1
    active_slots = torch.empty((B, T, O, model_SA.slots_dim), device=DEVICE)
    background_slots = torch.empty((B, T, 1, model_SA.slots_dim), device=DEVICE)
    active_attn = torch.empty((B, T, O, H*W), device=DEVICE)
    background_attn = torch.empty((B, T, 1, H*W), device=DEVICE)

    for t in range(T):
        # Encode / decode one frame
        slots_t, attn_t = model_SA.encode(obs[:, t], prev_slots)
        slots_t, attn_t = order_slots(slots_t, attn_t, prev_slots, prev_attn)
        
        active_slots[:, t] = slots_t[:, 1:]
        background_slots[:, t] = slots_t[:, :1]

        active_attn[:, t] = attn_t[:, 1:]
        background_attn[:, t] = attn_t[:, :1]

        prev_slots, prev_attn = slots_t, attn_t

    return active_slots, background_slots, active_attn, background_attn


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
    slots = slots.view(batch_size * seq_len, num_objects, slots_dim)
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