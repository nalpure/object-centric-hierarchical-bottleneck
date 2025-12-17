import os
from os import makedirs
from os.path import exists
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils import data

from explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from losses import attention_loss, disentanglement_loss, slot_slot_contrastive_loss
from slot_attention.autoencoder import SlotAttentionAutoEncoder, order_slots
from utils import reorder_perturbation_indices


class TrainStep:    
    def __init__(self, model : torch.nn.Module, device : torch.device):
        self.model = model
        self.device = device

    def __call__(self, batch) -> torch.Tensor:
        raise NotImplementedError("TrainStep is an abstract base class")
    

class SlotAttentionAETrainStep(TrainStep):
    def __init__(self, model: SlotAttentionAutoEncoder, device, recon_weight, bg_attn_weight):
        super().__init__(model, device)
        self.recon_weight = recon_weight
        self.bg_attn_weight = bg_attn_weight
        self.criterion = torch.nn.MSELoss()

    def __call__(self, batch) -> torch.Tensor:
        # Compute reconstruction
        obs = batch.to(self.device)
        recon_combined, recons, masks, attn = self.model(obs)

        # Compute losses
        loss_dict = {}
        recon_loss = self.criterion(obs, recon_combined) * self.recon_weight
        loss_dict["reconstruction"] = recon_loss

        if self.bg_attn_weight > 0.0:
            attn_loss = attention_loss(attn) * self.bg_attn_weight
            loss_dict["attention"] = attn_loss

        info_dict = {
            "orig": obs.detach().cpu(),
            "recon_combined": recon_combined.detach().cpu(),
            "recons": recons.detach().cpu(),
            "masks": masks.detach().cpu(),
            "attn": attn.detach().cpu(),
        }

        return loss_dict, info_dict
    

class SlotAttentionContrastiveTrainStep(TrainStep):
    def __init__(self, model: SlotAttentionAutoEncoder, device, recon_weight, bg_attn_weight, contrastive_weight):
        super().__init__(model, device)
        self.recon_weight = recon_weight
        self.bg_attn_weight = bg_attn_weight
        self.contrastive_weight = contrastive_weight
        self.criterion = torch.nn.MSELoss()

    def __call__(self, batch) -> torch.Tensor:
        img_seq = batch.to(self.device) # (B, T, C, H, W)
        B, T, C, H, W = img_seq.shape
        S = self.model.num_slots
        D = self.model.slots_dim
        O = S - 1

        # Encode all timesteps, initialized with previous slots
        active_slots = torch.empty((B, T, O, D), device=self.device)
        bg_slot = torch.empty((B, T, 1, D), device=self.device)
        attn = torch.empty((B, T, S, H*W), device=self.device)

        prev_slots = None
        prev_attn = None

        for t in range(T):
            slots_current, attn_current = self.model.encode(img_seq[:, t], slots_init=prev_slots)
            slots_current, attn_current = order_slots(slots_current, attn_current, prev_slots, prev_attn)

            active_slots[:, t] = slots_current[:, 1:]
            bg_slot[:, t] = slots_current[:, :1]
            attn[:, t] = attn_current

            prev_slots = slots_current
            prev_attn = attn_current

        # Decode all timesteps at once for speedup
        slots = torch.cat((active_slots, bg_slot), dim=2) # (B, T, S, D)
        slots_flat = slots.view(B * T, S, -1)
        recon_flat, _, _ = self.model.decode(slots_flat)
        recon = recon_flat.view(B, T, C, H, W)

        # Calculate losses
        loss_dict = {
            "reconstruction": self.criterion(recon, img_seq) * self.recon_weight,
            "contrastive": slot_slot_contrastive_loss(active_slots) * self.contrastive_weight,
            "attention": attention_loss(attn.view(B * T, S, H * W)) * self.bg_attn_weight
        }

        info_dict = {}
        
        return loss_dict, info_dict
    

class ExplicitAETrainStep(TrainStep):
    def __init__(self, model: ExplicitLatentAutoEncoder, device, recon_weight, disentangle_weight, noise_mag=0.0):
        super().__init__(model, device)
        self.recon_weight = recon_weight
        self.disentangle_weight = disentangle_weight
        self.noise_mag = noise_mag
        self.criterion = torch.nn.MSELoss()

    def __call__(self, batch) -> torch.Tensor:
        slots_orig, slots_pert, magnitude, obj, prop = (b.to(self.device) for b in batch)
        prop = reorder_perturbation_indices(prop)
        B, O, E = slots_orig.shape

        if self.noise_mag > 0.0:
            noise = torch.randn_like(slots_orig)
            slots_orig = slots_orig + noise * self.noise_mag
            slots_pert = slots_pert + noise * self.noise_mag
        
        slots_all = torch.cat([slots_orig, slots_pert], dim=0)  # [2*B, O, E]
        slot_recon_all, z = self.model(slots_all)
        z_orig, z_pert = z.split(B, dim=0)

        loss_dict = {
            "reconstruction": self.criterion(slots_all, slot_recon_all) * self.recon_weight
        }

        if self.disentangle_weight > 0.0:
            disent_loss = disentanglement_loss(z_orig, z_pert, latent_idx=prop, magnitude=magnitude)
            loss_dict["disentanglement"] = disent_loss * self.disentangle_weight

        info_dict = {}

        return loss_dict, info_dict
    

class ImplicitDynamicsTrainStep(TrainStep):
    def __init__(self, model, device, noise_mag, pred_loss_weight, disentangle_loss_weight, t_past, t_future):
        super().__init__(model, device)
        self.noise_mag = noise_mag
        self.pred_loss_weight = pred_loss_weight
        self.disentangle_loss_weight = disentangle_loss_weight
        self.t_past = t_past
        self.t_future = t_future

    def __call__(self, batch) -> torch.Tensor:
        criterion = torch.nn.MSELoss()
        disentangle = self.disentangle_loss_weight > 0.0
        orig_seq, pert_seq, magnitude, obj_index, prop_index = (a.to(self.device) for a in batch)
        prop_index = reorder_perturbation_indices(prop_index, shift=-3)
        B, _, O, E = orig_seq.shape
        orig_seq_past, orig_seq_future = orig_seq.split([self.t_past, self.t_future], dim=1)
        pert_seq_past, pert_seq_future = pert_seq.split([self.t_past, self.t_future], dim=1)

        if self.noise_mag > 0.0:
            noise = torch.randn_like(orig_seq_past) * self.noise_mag
            orig_seq_past = orig_seq_past + noise
            pert_seq_past = pert_seq_past + noise

        # Predict future explicit latents with implicit dynamics model
        seq_past_flat = torch.cat([orig_seq_past, pert_seq_past], dim=0).reshape(2*B, self.t_past, O, E)  # [2*B*T_past, O, D_slot]
        z_pred, z_implicit_first = self.model(seq_past_flat, self.t_future, disentangle=disentangle)
        seq_pred = z_pred[:, :, :, :E]
        orig_seq_pred, pert_seq_pred = seq_pred.split(B, dim=0)  # Each: [B, T_past + T_future, O, D_slot] or [B, T_future, O, D_slot]
        
        if z_implicit_first is not None:
            z_orig, z_pert = z_implicit_first.split(B, dim=0)

        # ===== COMPUTE LOSSES =====                
        pred_loss = (criterion(orig_seq_future, orig_seq_pred) + criterion(pert_seq_future, pert_seq_pred)) / 2.0
        loss_dict = {
            "prediction": pred_loss * self.pred_loss_weight
        }

        if disentangle:
            dis_loss = disentanglement_loss(z_orig, z_pert, latent_idx=prop_index, magnitude=magnitude)
            loss_dict["disentanglement"] = dis_loss * self.disentangle_loss_weight

        info_dict = {}

        return loss_dict, info_dict
    

class TrainManager:
    def __init__(self, train_step : TrainStep, dataloader : data.DataLoader, lr: float, warmup_epochs : int, decay_epochs: int, decay_rate: float, weight_decay: float):
        self.train_step = train_step
        self.dataloader = dataloader
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.weight_decay = weight_decay

        self.model = train_step.model
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = self._get_lr_schedule()
        self.epoch_idx = 0
        self.epoch_losses = {}
        self.best_epoch_idx = None
        self.best_loss = 1e9

    def _get_lr_schedule(self):
        """ Creates a learning rate scheduler with warmup and exponential decay."""
        def lr_lambda(current_step):
            if current_step < self.warmup_epochs:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_epochs))
            else:
                # Exponential decay after warmup
                decay_factor = (current_step - self.warmup_epochs) / self.decay_epochs
                return self.decay_rate ** decay_factor
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self):
        self.epoch_idx += 1
        self.epoch_loss_dict = {}

        for batch in self.dataloader:
            loss_dict, _ = self.train_step(batch)
            batch_loss = sum(loss_dict.values())

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            self._add_losses_to_epoch(loss_dict)

        self.scheduler.step()  # Adjust the learning rates
        self._normalize_epoch_losses()
        self._update_best()
        
    def save_checkpoint(self, ckpt_path: str, overwrite: bool = False):
        if exists(ckpt_path) and not overwrite:
            raise ValueError(f"Checkpoint path '{ckpt_path}' already exists. Will not overwrite.")

        ckpt_dir = os.path.dirname(ckpt_path)
        if not exists(ckpt_dir):
            makedirs(ckpt_dir)
            print(f"Created checkpoint directory at '{ckpt_dir}'.")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch_idx
        }, ckpt_path)

    def save_if_best(self, ckpt_path: str):
        if self.epoch_idx == self.best_epoch_idx:
            self.save_checkpoint(ckpt_path, overwrite=True)

    def log_losses(self):
        msg = f'Epoch #{self.epoch_idx}: '
        msg += ', '.join([f'{key}: {value:.6f}' for key, value in self.epoch_losses.items()])
        msg += f', lr: {self.scheduler.get_last_lr()[0]:.6f}'
        print(msg)

    def save_losses_to_csv(self, csv_path: str):
        file_exists = exists(csv_path)
        with open(csv_path, 'a') as f:
            if not file_exists:
                # Write header
                header = 'epoch,' + ','.join(self.epoch_losses.keys()) + '\n'
                f.write(header)
            # Write losses
            line = f"{self.epoch_idx}," + ','.join([f"{value}" for value in self.epoch_losses.values()]) + '\n'
            f.write(line)

    def _update_best(self):
        current_loss = sum(self.epoch_losses.values())
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch_idx = self.epoch_idx

    def _add_losses_to_epoch(self, loss_dict):
        for key, value in loss_dict.items():
            if key not in self.epoch_losses:
                self.epoch_losses[key] = 0.0
            self.epoch_losses[key] += value.item()
    
    def _normalize_epoch_losses(self):
        for key in self.epoch_losses:
            self.epoch_losses[key] /= len(self.dataloader)