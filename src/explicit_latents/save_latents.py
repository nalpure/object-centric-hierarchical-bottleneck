import os
import torch
from torch.utils import data
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, separate_slots
from src.utils import PerturbedImageSequenceDataset, load_config, get_config_argument, save_dict_h5py, set_seed, DEVICE, IMG_CHANNELS

NUM_WORKERS = 4  # for DataLoader

def main():
    config_name = get_config_argument()
    config = load_config(config_name)
    config_SA = config["slot_attention"]
    config_EL = config["explicit_latents"]
    obs_path = config_SA["train_path"]
    sa_ckpt_path = config_SA["ckpt_path"]
    expl_ckpt_path = config_EL["ckpt_path"]
    output_path = config_EL['save_path']

    if not os.path.exists(sa_ckpt_path):
        raise FileNotFoundError(f"Slot Attention Checkpoint path does not exist: {sa_ckpt_path}")
    if not os.path.exists(expl_ckpt_path):
        raise FileNotFoundError(f"Explicit Latents Checkpoint path does not exist: {expl_ckpt_path}")
    if not os.path.isdir(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        print(f"Created output directory: {os.path.dirname(output_path)}")
    
    print(f"Loading observation training dataset: {obs_path}")
    orig_dataset = PerturbedImageSequenceDataset(h5_path=obs_path, hdf5_format=config_SA["hdf5_format"])
    dataloader = data.DataLoader(orig_dataset, batch_size=config_SA['batch_size'], shuffle=False, drop_last=True, num_workers=NUM_WORKERS)
    print(f"Finished loading all {len(dataloader) * config_SA['batch_size']} observation samples.")

    print("Loading Slot Attention model:", sa_ckpt_path)
    sa_model = SlotAttentionAutoEncoder(
        resolution=config_SA["resolution"],
        num_slots=config_SA["num_slots"],
        num_iterations=config_SA["num_iterations"], 
        num_channels=IMG_CHANNELS,
        slots_dim=config_SA["slots_dim"], 
        encdec_dim=config_SA["encdec_dim"]).to(DEVICE)
    
    sa_model.load_state_dict(torch.load(sa_ckpt_path, weights_only=True)['model_state_dict'], strict=True)
    sa_model.eval()

    print("Loading Explicit Latents model:", expl_ckpt_path)
    expl_model = ExplicitLatentAutoEncoder(config_EL["latent_dim"], config_SA["slots_dim"]).to(DEVICE)
    expl_model.load_state_dict(torch.load(expl_ckpt_path, weights_only=True)['model_state_dict'], strict=True)
    expl_model.eval()

    print(f"Saving explicit latents to {output_path}")

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            obs_original, obs_perturbed, magnitude, obj_index, prop_index = batch
            seq_len = obs_original.size(1)
            obs_original = obs_original.to(DEVICE)
            obs_perturbed = obs_perturbed.to(DEVICE)

            num_objects = config_SA['num_slots'] - 1  # assuming one slot is reserved for background    
            orig_seq_latents = torch.empty((config_SA['batch_size'], seq_len, num_objects, config_EL['latent_dim'])).to(DEVICE)
            pert_seq_latents = torch.empty((config_SA['batch_size'], seq_len, num_objects, config_EL['latent_dim'])).to(DEVICE)

            slots_init_orig = None
            slots_init_pert = None
            for i in range(seq_len):
                slots_orig, attention_orig = sa_model.encode(obs_original[:, i], slots_init=slots_init_orig)
                slots_pert, attention_pert = sa_model.encode(obs_perturbed[:, i], slots_init=slots_init_pert)
                active_slots_orig, _ = separate_slots(slots_orig, attention_orig)
                active_slots_pert, _ = separate_slots(slots_pert, attention_pert)
                z_orig = expl_model.encode(active_slots_orig)
                z_pert = expl_model.encode(active_slots_pert)
                orig_seq_latents[:, i] = z_orig
                pert_seq_latents[:, i] = z_pert
                slots_init_orig = slots_orig
                slots_init_pert = slots_pert
           
            data_dict = {
                f'batch_{batch_index}_orig_seq': orig_seq_latents,
                f'batch_{batch_index}_pert_seq': pert_seq_latents,
                f'batch_{batch_index}_magnitude': magnitude,
                f'batch_{batch_index}_obj_index': obj_index,
                f'batch_{batch_index}_prop_index': prop_index
            }
            mode = 'w' if batch_index == 0 else 'a'
            save_dict_h5py(data_dict, output_path, mode)

            if batch_index == 0:
                for sample_index in range(min(3, config_SA['batch_size'])):
                    print(f"------- SAMPLE {sample_index} -------")
                    for t in range(seq_len):
                        print(f"--- time t={t} ---")
                        for obj in range(num_objects):
                            latent_str = ", ".join([f"{val:.3f}" for val in orig_seq_latents[sample_index, t, obj, :]])
                            print(f"obj {obj + 1}: [{latent_str}]")
                        print()
                    print()

    print("Finished saving all latents.")
    

if __name__ == "__main__":
    main()