import argparse
import os
import numpy as np
import torch
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

#from src.implicit_latents.autoencoder import ImplicitLatentAutoEncoder
from implicit_latents.relational_latent_dynamics import RelationalLatentDynamics
from src.utils import get_explicit_codes, load_config, load_config_by_name, save_config, set_seed, DEVICE
from datasets import PerturbedSlotSequenceDataset
from train_classes import ImplicitDynamicsTrainStep, TrainManager

T_PAST = 4
T_FUTURE = 4

print("Running on", DEVICE)


# ----- Load configuration -----

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Model and training configuration")
parser.add_argument("-n", "--name", help="Name for the training run.")
parser.add_argument("-d", "--data", help="Dataset path.")
parser.add_argument("-b", "--base", help="Base model name.")    
parser.add_argument(
    "-e",
    "--base-epoch",
    help="Base model epoch. If not provided, latest epoch is selected.",
)
args = parser.parse_args()
config = load_config_by_name(args.config)

if "seed" not in config:
    config["seed"] = np.random.randint(2**31)

set_seed(config['seed'])

if args.name is None:
    if not "name" in config:
        raise "Provide a name for the run!"
else:
    config["name"] = args.name

if args.data is None:
    if not "data_path" in config:
        raise "Provide a dataset path!"
else:
    config["data_path"] = args.data

if args.base is None:
    print("No base model specified.")
else:
    base_config = load_config(f"out/{args.base}/config.toml")
    config["type"] = base_config["type"]
    config["model"] = base_config["model"]
    config["base_model"] = args.base
    
if not args.base_epoch is None:
    config["base_epoch"] = args.base_epoch


# ----- Create output folder -----

if not os.path.exists("out"):
    os.mkdir("out")

if os.path.exists(f"out/{config['name']}"):
    run_index = 0
    while os.path.exists(f"out/{config['name']}_{run_index}"):
        run_index += 1
    run_name = f"{config['name']}_{run_index}"
else:
    run_name = config["name"]

print(f"Writing to output folder {run_name}...")
output_path = f"out/{run_name}"
os.mkdir(output_path)
save_config(config, f"{output_path}/config.toml")


# ----- Load dataset, model, and train classes -----

print("Loading training data...")
dataset = PerturbedSlotSequenceDataset(
    hdf5_file=config["data_path"], 
    normalize=False, 
    timesteps=T_PAST + T_FUTURE,
    prop_skip_codes=get_explicit_codes()
)

batch_size = config["train"]["batch_size"]
num_workers = config["train"]["num_workers"]
train_dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
print(f"Finished loading all {batch_size * len(train_dataloader)} training samples.")

explicit_dim = next(iter(train_dataloader))[0].shape[-1]
model = RelationalLatentDynamics(
    explicit_dim,
    config["dynamics"]["latent_dim"] - explicit_dim,
    T_PAST,
    config["dynamics"]["edge_dim"],
    config["dynamics"]["latent_edge_dim"]
).to(DEVICE)

init_ckpt = None # TODO !!
if init_ckpt is not None:
    ckpt = f"{init_ckpt}"
    print(f"Loading model weights from {ckpt}")
    checkpoint = torch.load(ckpt, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

train_step = ImplicitDynamicsTrainStep(
    model=model,
    device=DEVICE,
    loss_divisor=len(train_dataloader),
    noise_mag=0.0,
    pred_loss_weight=config["dynamics"]["weights"]["prediction"],
    disentangle_loss_weight=config["dynamics"]["weights"]["disentanglement"],
    t_past=T_PAST,
    t_future=T_FUTURE
)

train_manager = TrainManager(
    train_step=train_step,
    dataloader=train_dataloader,
    lr=config["train"]["opt"]["learning_rate"],
    warmup_epochs=config["train"]["opt"]["warmup_epochs"],
    decay_epochs=config["train"]["opt"]["decay_epochs"],
    decay_rate=config["train"]["opt"]["decay_rate"]
)


# ----- Train model -----

for epoch in tqdm(range(config["train"]["num_epochs"])):
    train_manager.train_epoch()
    train_manager.save_if_best(f"{output_path}/ckpt.pt") # TODO !!
    train_manager.save_losses_to_csv(f"{output_path}/losses.csv")


print("Finished.")