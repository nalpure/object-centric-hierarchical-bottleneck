import os
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils import data
import torch

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder, identify_background, order_slots
from src.utils import PerturbedImageSequenceDataset, PerturbedImageSequenceDataset, get_config_argument, load_config, plot_grid, set_seed, DEVICE, IMG_CHANNELS

DISPLAY_SAMPLES = 10
OUTPUT_DIR = "data/figures/"

def main():
    config_name = get_config_argument()
    config = load_config(config_name)["slot_attention"]
    set_seed(config["seed"])
    ckpt_path = config['ckpt_path']
    output_path = config['save_path']

    print("Loading model:", ckpt_path)
    model = SlotAttentionAutoEncoder(
        resolution=config["resolution"],
        num_slots=config["num_slots"],
        num_iterations=config["num_iterations"], 
        num_channels=IMG_CHANNELS,
        slots_dim=config["slots_dim"], 
        encdec_dim=config["encdec_dim"]).to(DEVICE)
    model.eval()
    
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)['model_state_dict'], strict=False)

    print(f"Loading observation training dataset: {config['train_path']}")
    dataset = PerturbedImageSequenceDataset(h5_path=config["train_path"], hdf5_format=config["hdf5_format"])
    dataloader = data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    print(f"Finished loading all {config['batch_size'] * len(dataloader)} sequences.")
    print(f"Saving slots to {output_path}")

    with torch.no_grad(), h5py.File(output_path, "w") as hf:
            for batch_index, batch in enumerate(tqdm(dataloader)):
                orig_seq, pert_seq, magnitude, obj_index, prop_index = batch
                orig_seq = orig_seq.to(DEVICE)
                pert_seq = pert_seq.to(DEVICE)
                num_objects = model.num_slots - 1
                B, T, C, H, W = orig_seq.shape
                
                active_slots_orig = torch.empty((B, T, num_objects, model.slots_dim), device=DEVICE)
                active_slots_pert = torch.empty((B, T, num_objects, model.slots_dim), device=DEVICE)
                bg_slot_orig = torch.empty((B, T, 1, model.slots_dim), device=DEVICE)

                prev_slots_orig = None
                prev_attn_orig = None
                prev_slots_pert = None
                prev_attn_pert = None

                for t in range(T):
                    slots_orig, attn_orig = model.encode(orig_seq[:, t], slots_init=prev_slots_orig)
                    slots_pert, attn_pert = model.encode(pert_seq[:, t], slots_init=prev_slots_pert)
                    slots_orig, attn_orig = order_slots(slots_orig, attn_orig, prev_slots_orig, prev_attn_orig)
                    slots_pert, attn_pert = order_slots(slots_pert, attn_pert, prev_slots_pert, prev_attn_pert)
                    active_slots_orig[:, t, :, :] = slots_orig[:, 1:, :]
                    active_slots_pert[:, t, :, :] = slots_pert[:, 1:, :]
                    bg_slot_orig[:, t, :, :] = slots_orig[:, :1, :]
                    prev_slots_orig = slots_orig
                    prev_attn_orig = attn_orig
                    prev_slots_pert = slots_pert
                    prev_attn_pert = attn_pert

                data_dict = {
                    f'batch_{batch_index}_orig_seq': active_slots_orig,
                    f'batch_{batch_index}_pert_seq': active_slots_pert,
                    f'batch_{batch_index}_magnitude': magnitude,
                    f'batch_{batch_index}_obj_index': obj_index,
                    f'batch_{batch_index}_prop_index': prop_index
                }
                for key, value in data_dict.items():
                    hf.create_dataset(key, data=value.cpu().numpy())

                if batch_index == 0:
                    for sample_idx in range(min(DISPLAY_SAMPLES, B)):
                        image_rows = np.empty((model.num_slots + 2, T), dtype=object)
                        row_titles = ["Original", "Reconstruction"] 
                        row_titles += [f"Mask {i+1}" for i in range(model.num_slots)]
                        column_titles = [f"t={t}" for t in range(T)]

                        for t in range(T):
                            active_slots_t = active_slots_orig[sample_idx][t].unsqueeze(0)
                            bg_slot_t = bg_slot_orig[sample_idx][t].unsqueeze(0)
                            slots_t = torch.cat([bg_slot_t, active_slots_t], dim=1)
                            recon_t, _, masks_t = model.decode(slots_t)
                            image_rows[0, t] = orig_seq[sample_idx, t].cpu()
                            image_rows[1, t] = recon_t[0].cpu()
                            for o in range(model.num_slots):
                                image_rows[o + 2, t] = masks_t[o].unsqueeze(0).cpu()
                
                        save_path = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}_latents.png")
                        plot_grid(image_rows, row_titles, column_titles, save_path)
                    print(f"Saved latent visualizations to {OUTPUT_DIR}")
    
    print("Finished saving slots.")
    

if __name__ == "__main__":
    main()