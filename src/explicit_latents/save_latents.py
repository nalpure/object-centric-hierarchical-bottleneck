import os
import numpy as np
import torch
from torch.utils import data
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import  PerturbedSlotSequenceDataset, load_config, get_config_argument, plot_grid, save_dict_h5py, set_seed, DEVICE, IMG_CHANNELS
from tqdm import tqdm


def main():
    config_name = get_config_argument()
    config = load_config(config_name)
    config_SA = config["slot_attention"]
    config_EL = config["explicit_latents"]
    train_path = config_EL["train_path"]
    sa_ckpt_path = config_SA["ckpt_path"]
    expl_ckpt_path = config_EL["ckpt_path"]
    output_path = config_EL['save_path']
    num_workers = config_EL['num_workers']

    if not os.path.exists(sa_ckpt_path):
        raise FileNotFoundError(f"Slot Attention Checkpoint path does not exist: {sa_ckpt_path}")
    if not os.path.exists(expl_ckpt_path):
        raise FileNotFoundError(f"Explicit Latents Checkpoint path does not exist: {expl_ckpt_path}")
    if not os.path.isdir(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        print(f"Created output directory: {os.path.dirname(output_path)}")
    
    print(f"Loading observation training dataset: {train_path}")
    orig_dataset = PerturbedSlotSequenceDataset(train_path, normalize=False)
    dataloader = data.DataLoader(orig_dataset, batch_size=config_SA['batch_size'], shuffle=False, drop_last=True, num_workers=num_workers)
    print(f"Finished loading all {len(dataloader) * config_SA['batch_size']} observation samples.")

    print("Loading Explicit Latents model:", expl_ckpt_path)
    expl_model = ExplicitLatentAutoEncoder(config_EL["latent_dim"], config_SA["slots_dim"]).to(DEVICE)
    expl_model.load_state_dict(torch.load(expl_ckpt_path, weights_only=True)['model_state_dict'], strict=True)
    expl_model.eval()

    print(f"Saving explicit latents to {output_path}")

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(dataloader)):
            slots_orig, slots_pert, magnitude, obj_index, prop_index = batch
            slots_orig = slots_orig.to(DEVICE)
            slots_pert = slots_pert.to(DEVICE)

            B, T, O, D_slot = slots_orig.shape
            D_latent = expl_model.latent_dim

            slots_orig = slots_orig.view(B * T, O, D_slot)
            slots_pert = slots_pert.view(B * T, O, D_slot)

            orig_latents = expl_model.encode(slots_orig)
            pert_latents = expl_model.encode(slots_pert)

            orig_latents = orig_latents.view(B, T, O, D_latent)
            pert_latents = pert_latents.view(B, T, O, D_latent)

            data_dict = {
                f'batch_{batch_index}_orig_seq': orig_latents.detach().cpu().numpy(),
                f'batch_{batch_index}_pert_seq': pert_latents.detach().cpu().numpy(),
                f'batch_{batch_index}_magnitude': magnitude,
                f'batch_{batch_index}_obj_index': obj_index,
                f'batch_{batch_index}_prop_index': prop_index
            }
            mode = 'w' if batch_index == 0 else 'a'
            save_dict_h5py(data_dict, output_path, mode)

            # Print some latents and visualize masks for the first batch
            if batch_index == 0:
                for sample_index in range(min(5, config_SA['batch_size'])):
                    print(f"------- SAMPLE {sample_index} -------")
                    for t in range(T):
                        print(f"--- time t={t} ---")
                        for i in range(O):
                            latent_str = ", ".join([f"{val:.3f}" for val in orig_latents[sample_index, t, i, :]])
                            print(f"obj {i + 1}: [{latent_str}]")
                        print()
                    print()

    print("Finished saving all latents.")
    

if __name__ == "__main__":
    main()