import argparse
import json
import torch
import utils
from datetime import datetime
import os
import wandb

from torch.utils import data

import modules
from utils import log_progress, set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of training epochs.')
parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='Learning rate.')
parser.add_argument('--seed', default=0, type=int, 
                    help='random seed')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--action_dim', type=int, default=4,
                    help='Dimensionality of action space.')
parser.add_argument('--num_objects', type=int, default=6,
                    help='Number of object slots in model.')
parser.add_argument('--num_feat', type=int, default=4,
                    help='Number of feature per object.')
parser.add_argument('--log-interval', type=int, default=20,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--dataset', type=str,
                    default='data/spriteworld_train.h5',
                    help='Path to replay buffer.')
parser.add_argument('--embodied', action='store_true')
parser.add_argument('--config', type=str, default=None,
                    help='Experiment name.')
parser.add_argument('--experiment', type=int, default=None,
                    help='Experiment number.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--OC-config', type=str, default=None,
                    help='configuration for the object-centric encoder.')
parser.add_argument('--init_ckpt', type=str, 
                    default=None,
                    help='pretrained weights for the object-centric encoder.')
parser.add_argument('--wandb_project', default=None, type=str, help='wandb project')
parser.add_argument('--ignore_action', action='store_true', default=False)

args = parser.parse_args()
args = vars(args)
if args["config"] is not None:
    args["ckpt_name"] = args["config"]
    with open("sswm_configs.json", "r") as config_file:
        configs = json.load(config_file)[args["config"]]
        config_file.close()
    for key, value in configs.items():
        try:
            args[key] = value
        except KeyError:
            exit(f"{key} is not a valid parameter")

    for k,v in args.items():
        print(f'{k}: {v}')
    print()

exp_name = args["config"]

save_folder = '{}/{}/'.format(args["save_folder"], exp_name)

if not os.path.exists(save_folder): os.makedirs(save_folder)
exp = 1 if args["experiment"] is None else args["experiment"]
model_file = os.path.join(save_folder, f'model_{exp}.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading state transitions dataset", args['dataset'])
dataset = utils.StateTransitionsDataset(hdf5_file=args["dataset"])
print("Loading data...")
train_loader = data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
print("Finished loading dataset.")

use_wandb = args["wandb_project"] is not None and args["wandb_entity"] is not None

if args["OC_config"] is not None:

    if args['init_ckpt']:
        print("Loading autoencoder checkpoint from:", args['init_ckpt'])
    else:
        print("Set up model for end-to-end training.")

    loss_name = "transition_loss"

    with open("configs.json", "r") as config_file:
        OC_configs = json.load(config_file)[args["OC_config"]]
        config_file.close()
    
    model = modules.SlotSWM(
        args=OC_configs, 
        hidden_dim=args["hidden_dim"],
        action_dim=args["action_dim"], 
        sigma=args["sigma"],
        embodied = True,
        init_weights=args["init_ckpt"],
        ignore_action=args["ignore_action"],
        device=device).to(device)

else:

    loss_name = "contrastive_loss"

    model = modules.ContrastiveSWM(
        embedding_dim=args["embedding_dim"],
        hidden_dim=args["hidden_dim"],
        action_dim=args["action_dim"],
        input_dims=tuple(args["input_dim"]),
        num_objects=args["num_objects"],
        num_feat=args["num_feat"],
        sigma=args["sigma"],
        hinge=args["hinge"],
        ignore_action=args["ignore_action"],
        copy_action=args["copy_action"],
        encoder=args["encoder"], 
        embodied=args["embodied"],
        device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

if use_wandb:
    wandb.init(project=args["wandb_project"], entity=args["wandb_entity"])
    wandb.run.name = args["config"]
    logs = {}
    for key, value in args.items():
        logs[key] = value
    wandb.config = logs
    wandb.watch(model)

set_seed(args['seed'])

# Train model.
print('Starting model training...')
step = 0
best_loss = 1e9
start = datetime.now()
model.train()

for epoch in range(1, args["epochs"] + 1):
    train_loss = 0

    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        obs, action, next_obs = data_batch
        #TODO make dynamic
        #obs = obs[:,:3,:,:]
        #next_obs = next_obs[:,:3,:,:]

        if args['ignore_action']:
            action = torch.zeros(obs.shape[0], 3, dtype=torch.int64, device=device)

        loss = model.contrastive_loss(obs, action, next_obs)
        
        if use_wandb: wandb.log({loss_name: loss}, step=step)
        
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

      
        if batch_idx > 0 and batch_idx % args["log_interval"] == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_batch[0]),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

        step += 1

    avg_loss = train_loss / len(train_loader)
    log_progress(epoch, args['epochs'], start, additional_msg=f'Average loss: {avg_loss:.6f}')
    print()

    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save(epoch, optimizer, model_file)

print(f"training time({args['epochs']} epochs): {(datetime.now() - start).total_seconds()}s")