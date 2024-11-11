import argparse
import torch
import utils
import json

from torch.utils import data
import numpy as np
from collections import defaultdict

import modules

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--num-steps', type=int, default=1,
                    help='Number of prediction steps to evaluate.')
parser.add_argument('--dataset', type=str,
                    default='data/spriteworld_test_0.h5',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--OC-config', type=str,
                    default=None,
                    help='Configuration used to instanciate a OC encoder (in config.json).')
parser.add_argument('--config', type=str,
                    default=None,
                    help='Configuration used to instanciate the baseline (in sswm_config.json).')
parser.add_argument('--num_output_figs', default=3, 
                    type=int, 
                    help='desired number of output figures')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_configuration(parser):
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

    dataset = args["dataset"]

    dataset = utils.PathDataset(hdf5_file=dataset, path_length=args["num_steps"])
    eval_loader = data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2, drop_last=True)

    if args["OC_config"] is not None:
        with open("configs.json", "r") as config_file:
            OC_configs = json.load(config_file)[args["OC_config"]]
            config_file.close()
    
        model = modules.SlotSWM(
        args=OC_configs, 
        hidden_dim=args["hidden_dim"],
        action_dim=args["action_dim"], 
        sigma=args["sigma"],
        ignore_action=args["ignore_action"],
        embodied = True,
        device=device).to(device)

    else:
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

    model.load_state_dict(torch.load(f"checkpoints/{args['config']}/model_1.pt", map_location=device)['model_state_dict'])
    model.eval()
    return args,eval_loader,model


def get_scores(pred_states, next_states, topk):
    hits_at = dict()
    rr_sum = 0
    num_samples = 0

    pred_state_cat = torch.cat(pred_states, dim=0)
    next_state_cat = torch.cat(next_states, dim=0)

    full_size = pred_state_cat.size(0)

    # Flatten object/feature dimensions
    next_state_flat = next_state_cat.view(full_size, -1)
    pred_state_flat = pred_state_cat.view(full_size, -1)

    dist_matrix = utils.pairwise_distance_matrix(next_state_flat, pred_state_flat)
    dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
    dist_matrix_augmented = torch.cat([dist_matrix_diag, dist_matrix], dim=1)

    # Workaround to get a stable sort in numpy.
    dist_np = dist_matrix_augmented.numpy()
    indices = []
    for row in dist_np:
        keys = (np.arange(len(row)), row)
        indices.append(np.lexsort(keys))
    indices = np.stack(indices, axis=0)
    indices = torch.from_numpy(indices).long()

    labels = torch.zeros(
        indices.size(0), device=indices.device,
        dtype=torch.int64).unsqueeze(-1)

    num_samples += full_size
    
    print('Size of current topk evaluation batch: {}'.format(
        full_size))

    for k in topk:
        match = indices[:, :k] == labels
        num_matches = match.sum()
        hits_at[k] = num_matches.item() / num_samples

    match = indices == labels
    _, ranks = match.max(1)

    reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
    rr_sum += reciprocal_ranks.sum()

    mrr = rr_sum / float(num_samples)

    return hits_at, mrr


def get_predictions(args, eval_loader, model):
    pred_states = []
    next_states = []

    true_obs = []
    true_next_obs = []
    pred_next_obs = []

    for data_batch in eval_loader:
        data_batch = [[t.to(device) for t in tensor] for tensor in data_batch]
        (obs, next_obs), actions = data_batch

        if args['ignore_action']:
            actions = torch.zeros(args['num_steps'], obs.shape[0], 2, dtype=torch.float64, device=device)
        
        #TODO make dynamic
        obs = obs[:,:3,:,:]
        next_obs = next_obs[:,:3,:,:]

        state = model(obs.to(device))
        next_state = model(next_obs.to(device))

        pred_state = state
        for i in range(args["num_steps"]):
            pred_state = model.transition_model(pred_state, actions[i][:,1])

        pred_states.append(pred_state.cpu())
        next_states.append(next_state.cpu())

        if len(true_obs) < args['num_output_figs']:
            recon_combined, recons, masks, slots = model.obj_encoder(obs)
            true_obs.append(obs)
            true_next_obs.append(next_obs)
            pred_next_obs.append(recon_combined)
    
    return pred_states, next_states


topk = [1,2,3]
args, eval_loader, model = setup_configuration(parser)

with torch.no_grad():    
    pred_states, next_states = get_predictions(args, eval_loader, model)
    hits_at, mrr = get_scores(pred_states, next_states, topk)

print('Processed {} batches of size {}'.format(
len(eval_loader), args["batch_size"]))

for k in topk:
    print('Hits @ {}: {}'.format(k, hits_at[k]))

print('MRR: {}'.format(mrr))