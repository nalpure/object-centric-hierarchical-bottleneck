from tqdm import tqdm
from torch.utils import data
import torch

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder
from src.utils import PerturbedImageSequenceDataset, PerturbedImageSequenceDataset, get_config_argument, load_config, save_dict_h5py, set_seed, DEVICE, IMG_CHANNELS


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
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    print(f"Finished loading all {config['batch_size'] * len(dataloader)} training samples.")
    print(f"Saving slots to {output_path}")

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(dataloader)):
            orig_seq, pert_seq, magnitude, obj_index, prop_index = batch
            seq_len = orig_seq.size(1)
            orig_seq = orig_seq.to(DEVICE)
            pert_seq = pert_seq.to(DEVICE)
            num_active_slots = config['num_slots'] - 1
            orig_seq_slots = torch.empty((config['batch_size'], seq_len, num_active_slots, config['slots_dim'])).to(DEVICE)
            pert_seq_slots = torch.empty((config['batch_size'], seq_len, num_active_slots, config['slots_dim'])).to(DEVICE)

            for i in range(seq_len):
                _, _, _, active_slots_original, _ = model(orig_seq[:, i])
                _, _, _, active_slots_perturbed, _ = model(pert_seq[:, i])
                orig_seq_slots[:, i] = active_slots_original
                pert_seq_slots[:, i] = active_slots_perturbed
           
            data_dict = {
                f'batch_{batch_index}_orig_seq': orig_seq_slots,
                f'batch_{batch_index}_pert_seq': pert_seq_slots,
                f'batch_{batch_index}_magnitude': magnitude,
                f'batch_{batch_index}_obj_index': obj_index,
                f'batch_{batch_index}_prop_index': prop_index
            }
            mode = 'w' if batch_index == 0 else 'a'
            save_dict_h5py(data_dict, output_path, mode)
    
    print("Finished saving slots.")
    

if __name__ == "__main__":
    main()