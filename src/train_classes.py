import os
from os import makedirs
from os.path import exists
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils import data

from losses import attention_loss, disentanglement_loss
from utils import reorder_perturbation_indices


class TrainStep:    
    def __init__(self, model : torch.nn.Module, device : torch.device, loss_divisor : int):
        self.model = model
        self.device = device
        self.loss_divisor = loss_divisor
        self.epoch_loss_dict = {"total": 0.0}

    def __call__(self, batch) -> torch.Tensor:
        raise NotImplementedError("TrainStep is an abstract base class")
    
    def get_losses(self) -> dict:
        return self.epoch_loss_dict
    
    def reset_losses(self):
        self.epoch_loss_dict = {}
    
    def _add_loss(self, name : str, value : torch.Tensor):
        if name not in self.epoch_loss_dict:
            self.epoch_loss_dict[name] = 0.0
        self.epoch_loss_dict[name] += value.item() / self.loss_divisor
    

class SlotAttentionAETrainStep(TrainStep):
    def __init__(self, model, device, loss_divisor, recon_weight, bg_attn_weight):
        super().__init__(model, device, loss_divisor)
        self.recon_weight = recon_weight
        self.bg_attn_weight = bg_attn_weight
        self.criterion = torch.nn.MSELoss()

    def __call__(self, batch) -> torch.Tensor:
        batch_loss = 0.0
        obs = batch.to(self.device)
        recon_combined, _, _, attn = self.model(obs)
        recon_loss = self.criterion(obs, recon_combined) * self.recon_weight
        batch_loss += recon_loss
        self._add_loss("reconstruction", recon_loss)

        if self.bg_attn_weight > 0.0:
            attn_loss = attention_loss(attn) * self.bg_attn_weight
            batch_loss += attn_loss
            self._add_loss("attention", attn_loss)

        self._add_loss("total", batch_loss)
        return batch_loss / self.loss_divisor
    

class ExplicitAETrainStep(TrainStep):
    def __init__(self, model, device, loss_divisor, recon_weight, disentangle_weight, noise_mag=0.0):
        super().__init__(model, device, loss_divisor)
        self.recon_weight = recon_weight
        self.disentangle_weight = disentangle_weight
        self.noise_mag = noise_mag
        self.criterion = torch.nn.MSELoss()

    def __call__(self, batch) -> torch.Tensor:
        batch_loss = 0.0
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

        recon_loss = self.criterion(slots_all, slot_recon_all) * self.recon_weight
        batch_loss += recon_loss
        self._add_loss("reconstruction", recon_loss)

        if self.disentangle_weight > 0.0:
            disent_loss = disentanglement_loss(z_orig, z_pert, latent_idx=prop, magnitude=magnitude)
            disent_loss = disent_loss * self.disentangle_weight
            batch_loss += disent_loss
            self._add_loss("disentanglement", disent_loss)

        self._add_loss("total", batch_loss)

        return batch_loss / self.loss_divisor
    

class ImplicitDynamicsTrainStep(TrainStep):
    def __init__(self, model, device, loss_divisor, noise_mag, pred_loss_weight, disentangle_loss_weight, t_past, t_future):
        super().__init__(model, device, loss_divisor)
        self.noise_mag = noise_mag
        self.pred_loss_weight = pred_loss_weight
        self.disentangle_loss_weight = disentangle_loss_weight
        self.t_past = t_past
        self.t_future = t_future

    def __call__(self, batch) -> torch.Tensor:
        batch_loss = 0.0
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
        pred_loss = pred_loss * self.pred_loss_weight
        self._add_loss("prediction", pred_loss)
        batch_loss += pred_loss

        if disentangle:
            dis_loss = disentanglement_loss(z_orig, z_pert, latent_idx=prop_index, magnitude=magnitude)
            dis_loss = dis_loss * self.disentangle_loss_weight
            self._add_loss("disentanglement", dis_loss)
            batch_loss += dis_loss

        self._add_loss("total", batch_loss)

        return batch_loss / self.loss_divisor
    

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
        self.current_epoch = 0
        self.best_epoch = None
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
        self.train_step.reset_losses()
        self.current_epoch += 1

        for batch in self.dataloader:
            batch_loss = self.train_step(batch)
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        self.scheduler.step()  # Adjust the learning rates
        epoch_loss = self.train_step.get_losses()["total"]

        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_epoch = self.current_epoch
        
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
            "epoch": self.current_epoch
        }, ckpt_path)

    def save_if_best(self, ckpt_path: str):
        if self.current_epoch == self.best_epoch:
            self.save_checkpoint(ckpt_path, overwrite=True)

    def log_losses(self):
        loss_dict = self.train_step.get_losses()
        msg = f'Epoch #{self.current_epoch}: '
        msg += ', '.join([f'{key}: {value:.6f}' for key, value in loss_dict.items()])
        msg += f', lr: {self.scheduler.get_last_lr()[0]:.6f}'
        print(msg)

    def save_losses_to_csv(self, csv_path: str):
        loss_dict = self.train_step.get_losses()
        file_exists = exists(csv_path)
        with open(csv_path, 'a') as f:
            if not file_exists:
                # Write header
                header = 'epoch,' + ','.join(loss_dict.keys()) + '\n'
                f.write(header)
            # Write losses
            line = f"{self.current_epoch}," + ','.join([f"{value}" for value in loss_dict.values()]) + '\n'
            f.write(line)

