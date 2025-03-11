import argparse
from utils import ObservationDataset, PerturbationDataset, set_seed
from torch.utils import data
import torch
from slot_attention.AE import SlotAttentionAutoEncoder
import json
import h5py
import os


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--config', default=None, type=str, help='name of the configuration to use')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--train_path', default='data/slipscape/train_data', type=str, help='Path to the training data')
    parser.add_argument('--ckpt_path', default='checkpoints/slipscape/', type=str, help='where the model is saved')
    parser.add_argument('--output_dir', default='data/generated_slots/')
    parser.add_argument('--batch_size', default=64, type=int)

    # Image parameters
    parser.add_argument('--hdf5_format', default='CHW', type=str, help='format of train, val and test data frames')
    parser.add_argument('--resolution', default=[64, 64], type=list)
    parser.add_argument('--stacked_frames', default=1, type=int, help='number of frames stacked in each sample')
    parser.add_argument('--channels_per_frame', default=3, type=int, help='number of channels for a single frame')

    # Further Slot Attention parameters
    parser.add_argument('--num_slots', default=4, type=int, help='Number of slots in Slot Attention')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations')
    parser.add_argument('--slots_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--encdec_dim', default=32, type=int, help='encoder/decoder dimension size') 

    args = parser.parse_args()
    args = vars(args)

    if args["config"] is not None:
        args["ckpt_name"] = args["config"]
        with open("configs.json", "r") as config_file:
            configs = json.load(config_file)[args["config"]]
        for key, value in configs.items():
            try:
                args[key] = value
            except KeyError:
                Warning(f"{key} is not a valid parameter")

    return args


def main():
    args = parse_arguments()
    set_seed(args["seed"])
    ckpt_path = f"{args['ckpt_path']}{args['ckpt_name']}.ckpt"
    output_path = f"{args['output_dir']}{args['ckpt_name']}_slots.h5" 

    print("Loading model:", ckpt_path)
    model = SlotAttentionAutoEncoder(
        resolution=args["resolution"],
        num_slots=args["num_slots"],
        num_frames = args["stacked_frames"],
        num_iterations=args["num_iterations"], 
        num_channels=args["channels_per_frame"],
        slots_dim=args["slots_dim"], 
        encdec_dim=args["encdec_dim"]).to(DEVICE)
    model.eval()
    
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path, weights_only=True)['model_state_dict'], strict=False)
    # these keys are not in the checkpoint but will be generated again
    missing_keys.remove('encoder_cnn.encoder_pos.grid') 
    missing_keys.remove('decoder_cnn.decoder_pos.grid') 
    
    if missing_keys:
        raise KeyError(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        raise KeyError(f"Unexpected keys: {unexpected_keys}")

    print(f"Loading observation training dataset: {args['train_path']}")
    test_dataset = PerturbationDataset(hdf5_file=args["train_path"], hdf5_format=args["hdf5_format"])
    train_dataloader = data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    print(f"Finished loading all {args['batch_size'] * len(train_dataloader)} training samples.")
    print(f"Saving slots to {output_path}")

    with torch.no_grad():
        for batch_index, batch in enumerate(train_dataloader):
            obs, perturbed, _, _, _ = batch
            obs = obs.to(DEVICE)
            perturbed = perturbed.to(DEVICE)
            _, _, _, active_slots_original, _ = model(obs)
            _, _, _, active_slots_perturbed, _ = model(perturbed)
            save_slots_to_hdf5(active_slots_original, active_slots_perturbed, output_path, batch_index)
    
    print("Finished saving slots.")


def save_slots_to_hdf5(slots_original, slots_perturbed, output_path, batch_index):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create new file if batch_index is 0, otherwise append to existing file
    mode = 'w' if batch_index == 0 else 'a'
    with h5py.File(output_path, mode) as f:
        dset_name_original = f'batch_{batch_index}_original'
        dset_name_perturbed = f'batch_{batch_index}_perturbed'
        f.create_dataset(dset_name_original, data=slots_original.cpu().numpy(), compression="gzip", chunks=True)
        f.create_dataset(dset_name_perturbed, data=slots_perturbed.cpu().numpy(), compression="gzip", chunks=True)


if __name__ == "__main__":
    main()