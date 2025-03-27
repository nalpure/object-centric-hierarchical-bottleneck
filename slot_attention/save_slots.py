from tqdm import tqdm
from utils import PerturbedImageSequenceDataset, get_config_argument, load_config, set_seed, DEVICE, IMG_CHANNELS
from torch.utils import data
import torch
from slot_attention.autoencoder import SlotAttentionAutoEncoder
import h5py
import os


def main():
    config_name = get_config_argument()
    config = load_config(config_name)["slot_attention"]
    set_seed(config["seed"])
    ckpt_path = config['ckpt_path']
    output_path = config['slot_save_path']

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
    orig_dataset = PerturbedImageSequenceDataset(hdf5_file=config["train_path"], hdf5_format=config["hdf5_format"])
    dataloader = data.DataLoader(orig_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
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
                'orig_seq': orig_seq_slots,
                'pert_seq': pert_seq_slots,
                'magnitude': magnitude,
                'obj_index': obj_index,
                'prop_index': prop_index
            }
            save_slots_to_hdf5(data_dict, output_path, batch_index)
    
    print("Finished saving slots.")


def save_slots_to_hdf5(data_dict, output_path, batch_index):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create new file if batch_index is 0, otherwise append to existing file
    mode = 'w' if batch_index == 0 else 'a'
    with h5py.File(output_path, mode) as f:
        for key, value in data_dict.items():
            dset_name = f'batch_{batch_index}_{key}'
            f.create_dataset(dset_name, data=value.cpu().numpy(), compression="gzip", chunks=True)


if __name__ == "__main__":
    main()