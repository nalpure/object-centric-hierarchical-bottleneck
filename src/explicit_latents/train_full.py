import os
from os import makedirs
from os.path import exists
from datetime import datetime
import torch
from torch import optim
from torch.utils import data
from torch.nn import MSELoss

from implicit_latents.relational_latent_dynamics import RelationalLatentDynamics
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import PerturbedSlotSequenceDataset, get_config_argument, get_lr_schedule, load_config, log_progress, reorder_perturbation_indices, set_seed, DEVICE
from src.losses import disentanglement_loss

criterion = MSELoss()
verbose = True
EPS = 1e-6

T_past = 4 #TODO: make point of splitting sequences configurable
T_future = 4

print("Running on", DEVICE)
config_name = get_config_argument()
config = load_config(config_name)["explicit_latents"]

noise = config["noise"]
disentangle = config["disentanglement_loss_weight"] > 0.0
reconstruct = config["reconstruction_loss_weight"] > 0.0
T_future = T_future if config["prediction_loss_weight"] > 0.0 else 0
predict = T_future > 0

if exists(config["ckpt_path"]):
    raise ValueError(f"Checkpoint path '{config['ckpt_path']}' already exists. Please provide a new path to save the model checkpoint.")

for key, value in config.items():
    print(f"{key}: {value}")
set_seed(config['seed'])

mean = None
std = None
norm_stats_path = config["train_path"].replace(".h5", "_norm_stats.pt")

if config["normalize"] and exists(norm_stats_path):
    print("Loading normalization stats from", norm_stats_path)
    stats = torch.load(norm_stats_path)
    mean = stats["mean"]
    std = stats["std"]
    print(f"mean: {mean}, std: {std}")


print("Loading training data...")
dataset = PerturbedSlotSequenceDataset(hdf5_file=config["train_path"], feature_mean=mean, feature_std=std, normalize=config["normalize"])

if config["normalize"] and not exists(norm_stats_path):
    print(f"mean: {dataset.feature_mean}, std: {dataset.feature_std}")
    print("Saving normalization stats to", norm_stats_path)
    torch.save({"mean": dataset.feature_mean, "std": dataset.feature_std}, norm_stats_path)

train_dataloader = data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=config["num_workers"], prefetch_factor=8)
print(f"Finished loading all {len(train_dataloader)}x{config['batch_size']} training samples.")

slots_dim = dataset[0][0].shape[-1]

# Initialize the model
model_AE = ExplicitLatentAutoEncoder(
    config["explicit_dim"],
    slots_dim
).to(DEVICE)

model_implicit = RelationalLatentDynamics(
    explicit_dim=config["explicit_dim"],
    implicit_dim=config["implicit_dim"],
    seq_len=T_past,
    edge_dim=config["edge_dim"],
    latent_edge_dim=config["latent_edge_dim"],

).to(DEVICE)

if config["init_ckpt"] is not None:
    ckpt = f"{config['init_ckpt']}"
    if not ckpt.endswith(".ckpt"):
        ckpt += ".ckpt"
    print(f"Loading model weights from {ckpt}")
    checkpoint = torch.load(ckpt, weights_only=True)

    if "model_implicit_state_dict" in checkpoint:
        model_implicit.load_state_dict(checkpoint["model_implicit_state_dict"], strict=True)
        model_AE.load_state_dict(checkpoint["model_AE_state_dict"], strict=True)
    else:
        model_AE.load_state_dict(checkpoint["model_state_dict"], strict=True)

if config["freeze_ae"]:
    print("Freezing autoencoder weights.")
    for param in model_AE.parameters():
        param.requires_grad = False

# Initialize the optimizer for the model
trainable_params = [p for p in model_AE.parameters() if p.requires_grad] \
                 + [p for p in model_implicit.parameters() if p.requires_grad]

lr = config["learning_rate"]
if config["optimizer"] == "adam":
    optimizer = optim.Adam(trainable_params, lr=lr)
elif config["optimizer"] == "rmsprop":
    optimizer = optim.RMSprop(trainable_params, lr=lr)
elif config["optimizer"] == "sgd":
    optimizer = optim.SGD(trainable_params, lr=lr)
else:
    raise ValueError("Select a valid optimizer.")

print(f"Number of trainable autoencoder parameters: {sum(p.numel() for p in model_AE.parameters() if p.requires_grad)}")
print(f"Number of trainable implicit model parameters: {sum(p.numel() for p in model_implicit.parameters() if p.requires_grad)}")
ckpt_path = config["ckpt_path"]
ckpt_dir = os.path.dirname(ckpt_path)

if not exists(ckpt_dir):
    makedirs(ckpt_dir)

scheduler = get_lr_schedule(
    optimizer, 
    config["warmup_epochs"], 
    config["decay_epochs"], 
    config["decay_rate"]
)

best_loss = 1e9
model_AE.train()
model_implicit.train()
start = datetime.now()

print("Training started at", start.ctime())

for epoch in range(1, config["num_epochs"] + 1): 
    reconstruction_epoch_loss = 0
    recon_slot_epoch_loss = 0
    recon_latent_epoch_loss = 0
    prediction_epoch_loss = 0
    disentanglement_epoch_loss = 0
    epoch_loss = 0
    latent_std = 0

    for i, batch in enumerate(train_dataloader):
        orig_seq, pert_seq, magnitude, obj_index, prop_index = (b.to(DEVICE) for b in batch)
        prop_index = reorder_perturbation_indices(prop_index)

        B, T, O, S = orig_seq.shape
        T_out = T_future + T_past if reconstruct else T_future
        
        orig_seq_past = orig_seq[:, :T_past, :, :]
        pert_seq_past = pert_seq[:, :T_past, :, :]
        orig_seq_future = orig_seq[:, T_past:T_past+T_future, :, :]
        pert_seq_future = pert_seq[:, T_past:T_past+T_future, :, :]
        batch_loss = 0

        if noise > 0.0:
            orig_seq_past = orig_seq_past + torch.randn_like(orig_seq_past) * noise
            pert_seq_past = pert_seq_past + torch.randn_like(pert_seq_past) * noise
        
        # Encode to explicit latents
        seq_past_flat = torch.cat([orig_seq_past, pert_seq_past], dim=0).reshape(2*B*T_past*O, S)
        z_expl_past = model_AE.encode(seq_past_flat).reshape(2*B, T_past, O, -1)
        
        # Predict future explicit latents with implicit dynamics model
        z_expl_computed, z = model_implicit(z_expl_past, T_future, reconstruct=reconstruct)
        z_orig = z[:B]
        z_pert = z[B:]

        # Decode explicit latents back to slots
        z_expl_computed_flat = z_expl_computed.reshape(2*B*T_out*O, -1)
        seq_computed = model_AE.decode(z_expl_computed_flat).reshape(2, B, T_out, O, S)
        orig_seq_computed = seq_computed[0]
        pert_seq_computed = seq_computed[1]

        if reconstruct:
            # implicit latent dynamics
            orig_seq_recon = orig_seq_computed[:, :T_past, :, :]
            pert_seq_recon = pert_seq_computed[:, :T_past, :, :]
            recon_loss = criterion(orig_seq_past, orig_seq_recon) + criterion(pert_seq_past, pert_seq_recon)
            recon_loss *= config["reconstruction_loss_weight"] / 2.0
            batch_loss += recon_loss
            reconstruction_epoch_loss += recon_loss.item()

            # pure explicit AE reconstruction loss
            z_expl_past_flat = z_expl_past.reshape(2*B*T_past*O, -1)
            seq_computed_explicit = model_AE.decode(z_expl_past_flat).reshape(2, B, T_past, O, S)
            recon_slot_loss = criterion(orig_seq_past, seq_computed_explicit[0]) + criterion(pert_seq_past, seq_computed_explicit[1])
            recon_slot_loss *= config["reconstruction_loss_weight"] / 2.0
            recon_slot_epoch_loss += recon_slot_loss.item()

            # pure latent reconstruction loss
            z_expl_computed_past_flat = z_expl_computed[:, :T_past].reshape(2*B*T_past*O, -1)
            recon_latent_loss = criterion(z_expl_past, z_expl_computed[:, :T_past]) * 100
            recon_latent_epoch_loss += recon_latent_loss.item()
            batch_loss += recon_latent_loss
            latent_std += z_expl_past.std().item()

        if predict:
            orig_seq_pred = orig_seq_computed[:, T_past*reconstruct:T_out, :, :]
            pert_seq_pred = pert_seq_computed[:, T_past*reconstruct:T_out, :, :]
            pred_loss = criterion(orig_seq_future, orig_seq_pred) + criterion(pert_seq_future, pert_seq_pred)
            pred_loss *= config["prediction_loss_weight"] / 2.0
            batch_loss += pred_loss
            prediction_epoch_loss += pred_loss.item()
        
        if disentangle:
            disentangle_loss = disentanglement_loss(z_orig, z_pert, prop_index, magnitude)
            disentangle_loss *= config["disentanglement_loss_weight"]
            batch_loss += disentangle_loss
            disentanglement_epoch_loss += disentangle_loss.item()

        # Backpropagation and optimization step
        epoch_loss += batch_loss.item()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    
    scheduler.step() # Adjust the learning rates
    current_lr = scheduler.get_last_lr()
    loss_dict = {}
    
    loss_dict = {}
    if reconstruct:
        loss_dict["Recon Loss"] = reconstruction_epoch_loss / len(train_dataloader)
        loss_dict["Recon Explicit AE Loss"] = recon_slot_epoch_loss / len(train_dataloader)
        loss_dict["Recon Latent Loss"] = recon_latent_epoch_loss / len(train_dataloader)
    if predict:
        loss_dict["Prediction Loss"] = prediction_epoch_loss / len(train_dataloader)
    if disentangle:
        loss_dict["Disentanglement Loss"] = disentanglement_epoch_loss / len(train_dataloader)
    
    epoch_loss /= len(train_dataloader)

    # Save the best model and optimizer state
    marker = ""
    if epoch_loss < best_loss:
        marker = "*"
        best_loss = epoch_loss
        best_epoch = epoch

        torch.save({
            "model_AE_state_dict": model_AE.state_dict(),
            "model_implicit_state_dict": model_implicit.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }, ckpt_path)

    # Log progress
    if verbose:
        additional_msg = ", ".join([f"{key}: {value:.6f}" for key, value in loss_dict.items()])
        additional_msg += f", Total loss: {epoch_loss:.6f}{marker}, lr: {current_lr[0]:.6f}"
        additional_msg += f", Latent std: {latent_std / len(train_dataloader):.6f}"
        log_progress(epoch, config["num_epochs"], start, additional_msg)


if verbose:
    print()
    print(f"Best epoch was #{best_epoch} with a loss of {best_loss:.6f}. Saved at '{ckpt_path}'.")