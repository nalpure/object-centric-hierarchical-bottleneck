import argparse
from slot_attention.slot_attention import SlotAttentionAutoEncoder
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
from os import makedirs
from os.path import exists
import json
from torch.utils import data
from utils import StateTransitionsDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default=None, type=str, help='name of the configuration to use')
    parser.add_argument('--init_ckpt', default=None, type=str, help='initial weights to start training')
    parser.add_argument('--ckpt_path', default='checkpoints/spriteworld/', type=str, help='where to save models')
    parser.add_argument('--ckpt_name', default='model', type=str, help='where to save models')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--data_path', default='spriteworld/spriteworld/data', type=str, help='Path to the data')
    parser.add_argument('--resolution', default=[35, 35], type=list)
    parser.add_argument('--data_size', default=6400, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use. Choose between [adam, sgd, rmsprop, adamw, radam]')
    parser.add_argument('--small_arch', action='store_true', help='if true set the encoder/decoder dim to 32, 64 otherwise')
    parser.add_argument('--num_slots', default=4, type=int, help='Number of slots in Slot Attention')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations')
    parser.add_argument('--slots_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--learning_rate', default=0.0004, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers for loading data')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--wandb_project', default=None, type=str, help='wandb project')
    parser.add_argument('--wandb_entity', default=None, type=str, help='wandb entity')

    args = parser.parse_args()
    return vars(args)


def load_configuration(args):
    """Load configuration from a JSON file if provided."""
    if args["config"] is not None:
        args["ckpt_name"] = args["config"]
        with open("configs.json", "r") as config_file:
            configs = json.load(config_file)[args["config"]]
        for key, value in configs.items():
            if key in args:
                args[key] = value
            else:
                raise KeyError(f"{key} is not a valid parameter")
    return args


def setup_wandb(args, model):
    """Initialize Weights & Biases if required."""
    if args["wandb_project"] and args["wandb_entity"]:
        import wandb
        wandb.init(project=args["wandb_project"], entity=args["wandb_entity"])
        wandb.run.name = args["config"]
        wandb.config.update(args)
        wandb.watch(model)


def initialize_model(args):
    """Initialize the SlotAttentionAutoEncoder model."""
    model = SlotAttentionAutoEncoder(
        tuple(args["resolution"]),
        args["num_slots"],
        args["num_iterations"],
        args["slots_dim"],
        32 if args["small_arch"] else 64,
        args["small_arch"]
    ).to(device)

    model.encoder_cnn.encoder_pos.grid = model.encoder_cnn.encoder_pos.grid.to(device)
    model.decoder_cnn.decoder_pos.grid = model.decoder_cnn.decoder_pos.grid.to(device)

    # Load model weights from checkpoint if provided
    if args["init_ckpt"] is not None:
        checkpoint = torch.load(args["ckpt_path"] + args["init_ckpt"] + ".ckpt")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model


def initialize_optimizer(args, model):
    """Initialize optimizer based on provided configuration."""
    params = [{'params': model.parameters()}]
    if args["optimizer"] == "adam":
        optimizer = optim.Adam(params, lr=args["learning_rate"])
    elif args["optimizer"] == "rmsprop":
        optimizer = optim.RMSprop(params, lr=args["learning_rate"])
    elif args["optimizer"] == "sgd":
        optimizer = optim.SGD(params, lr=args["learning_rate"])
    else:
        raise ValueError("Select a valid optimizer.")
    
    return optimizer


def adjust_learning_rate(args, current_step, optimizer):
    """Adjust the learning rate based on warmup and decay schedule."""
    if current_step < args["warmup_steps"]:
        learning_rate = args["learning_rate"] * (current_step / args["warmup_steps"])
    else:
        learning_rate = args["learning_rate"]
    learning_rate = learning_rate * (args["decay_rate"] ** (current_step / args["decay_steps"]))
    optimizer.param_groups[0]['lr'] = learning_rate


def load_data(args):
    """Load dataset and create dataloaders."""
    dataset = StateTransitionsDataset(hdf5_file=args["data_path"])
    dataloader = data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])
    return dataloader


def train(model, optimizer, criterion, train_dataloader, args):
    """Main training loop."""
    model.train()
    start = time.time()
    current_step = 0
    for epoch in tqdm(range(args["num_epochs"])):
        total_loss = 0
        for sample in train_dataloader:
            adjust_learning_rate(args, current_step, optimizer)
            obs, action, next_obs = sample
            obs = preprocess_observations(obs, num_frames=1, channels_per_frame=3)
            obs = obs.to(device)

            recon_combined, recons, masks, slots = model(obs)
            loss = criterion(recon_combined, obs)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_step += 1

        total_loss /= len(train_dataloader)
        print(f"Epoch: {epoch}, Loss: {total_loss}, Time: {datetime.timedelta(seconds=time.time() - start)}")

        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, epoch, current_step, args)


def preprocess_observations(obs, num_frames, channels_per_frame):
    """Add noise to observations and return them."""
    obs = obs[:, :num_frames * channels_per_frame, :, :]
    noise = (torch.randint(0, 3, (1,)) > 0) * 0.5 * torch.rand((1, obs.shape[1], 1, 1)).clip(0, 1)
    obs = obs + noise
    return obs


def save_checkpoint(model, optimizer, epoch, current_step, args):
    """Save the model and optimizer state."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "epoch": (epoch, current_step)
    }, f"{args['ckpt_path']}{args['ckpt_name']}_{epoch}ep.ckpt")


def main():
    args = parse_arguments()
    args = load_configuration(args)

    if not exists(args["ckpt_path"]):
        makedirs(args["ckpt_path"])

    model = initialize_model(args)
    optimizer = initialize_optimizer(args, model)
    criterion = torch.nn.MSELoss()

    if args["init_ckpt"] is not None:
        checkpoint = torch.load(args["ckpt_path"] + args["init_ckpt"] + ".ckpt")
        try:
            optimizer.load_state_dict(checkpoint["optim_state_dict"])
        except Exception:
            pass

    setup_wandb(args, model)

    train_dataloader = load_data(args)
    train(model, optimizer, criterion, train_dataloader, args)


if __name__ == "__main__":
    main()