from tqdm import tqdm
from explicit_latents.autoencoder import LatentAutoEncoder
from utils import PerturbedSlotSequenceDataset, get_config_argument, load_config, save_dict_h5py, set_seed, DEVICE
from torch.utils import data
import torch


def main():
    config_name = get_config_argument()
    config = load_config(config_name)["explicit_latents"]
    set_seed(config["seed"])
    ckpt_path = config['ckpt_path']
    output_path = config['save_path']

    print(f"Loading observation training dataset: {config['train_path']}")
    orig_dataset = PerturbedSlotSequenceDataset(hdf5_file=config["train_path"])
    dataloader = data.DataLoader(orig_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
    print(f"Finished loading all {config['batch_size'] * len(dataloader)} training samples.")
    slots_dim = next(iter(dataloader))[0].shape[-1]

    print("Loading model:", ckpt_path)
    model = LatentAutoEncoder(config["latent_dim"], slots_dim).to(DEVICE)
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)['model_state_dict'], strict=True)

    print(f"Saving slots to {output_path}")

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(dataloader)):
            slots_original, slots_perturbed, magnitude, obj_index, prop_index = batch
            seq_len = slots_original.size(1)
            slots_original = slots_original.to(DEVICE)
            slots_perturbed = slots_perturbed.to(DEVICE)

            num_objects = slots_original.shape[2]     
            orig_seq_latents = torch.empty((config['batch_size'], seq_len, num_objects, config['latent_dim'])).to(DEVICE)
            pert_seq_latents = torch.empty((config['batch_size'], seq_len, num_objects, config['latent_dim'])).to(DEVICE)

            for i in range(seq_len):
                z_orig = model.encode(slots_original[:, i])
                z_pert = model.encode(slots_perturbed[:, i])
                orig_seq_latents[:, i] = z_orig
                pert_seq_latents[:, i] = z_pert
           
            data_dict = {
                f'batch_{batch_index}_orig_seq': orig_seq_latents,
                f'batch_{batch_index}_pert_seq': pert_seq_latents,
                f'batch_{batch_index}_magnitude': magnitude,
                f'batch_{batch_index}_obj_index': obj_index,
                f'batch_{batch_index}_prop_index': prop_index
            }
            mode = 'w' if batch_index == 0 else 'a'
            save_dict_h5py(data_dict, output_path, mode)
    
    print("Finished saving slots.")


if __name__ == "__main__":
    main()