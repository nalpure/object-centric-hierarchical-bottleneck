import h5py
from tqdm import tqdm
from torch.utils import data
import torch

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder
from src.utils import PerturbedImageSequenceDataset, PerturbedImageSequenceDataset, get_config_argument, load_config, set_seed, DEVICE, IMG_CHANNELS


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
    
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path, weights_only=True)['model_state_dict'], strict=False)
    
    # these keys can be generated again by the model
    generatable_keys = ['encoder_cnn.encoder_pos.grid', 'decoder_cnn.decoder_pos.grid']
    for key in generatable_keys:
        if key in missing_keys:
            missing_keys.remove(key)
    
    if missing_keys:
        raise KeyError(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        raise KeyError(f"Unexpected keys: {unexpected_keys}")

    print(f"Loading observation training dataset: {config['train_path']}")
    dataset = PerturbedImageSequenceDataset(h5_path=config["train_path"], hdf5_format=config["hdf5_format"])
    dataloader = data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=8,
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
                num_active_slots = config['num_slots'] - 1

                B, T, C, H, W = orig_seq.shape  # B=64, T=4
                orig_seq_flat = orig_seq.reshape(B * T, C, H, W)  # (256, 3, 64, 64)
                pert_seq_flat = pert_seq.reshape(B * T, C, H, W)

                # Single forward pass for all timesteps
                _, _, _, active_slots_original, _ = model(orig_seq_flat)
                _, _, _, active_slots_perturbed, _ = model(pert_seq_flat)

                # Reshape back to (B, T, num_slots, slot_dim)
                active_slots_original = active_slots_original.view(B, T, num_active_slots, active_slots_original.size(-1))
                active_slots_perturbed = active_slots_perturbed.view(B, T, num_active_slots, active_slots_perturbed.size(-1))

                data_dict = {
                    f'batch_{batch_index}_orig_seq': active_slots_original,
                    f'batch_{batch_index}_pert_seq': active_slots_perturbed,
                    f'batch_{batch_index}_magnitude': magnitude,
                    f'batch_{batch_index}_obj_index': obj_index,
                    f'batch_{batch_index}_prop_index': prop_index
                }
                for key, value in data_dict.items():
                    hf.create_dataset(key, data=value.cpu().numpy())
    
    print("Finished saving slots.")
    

if __name__ == "__main__":
    main()