from typing import List
import h5py
import numpy as np
import torch
from torch.utils import data


def transpose_array(arr, in_fmt: str, out_fmt: str):
    """
    Handles conversion between 'HWC' and 'CHW' formats for both
    single frames and sequences.
    """
    if in_fmt == out_fmt:
        return arr

    if in_fmt == "HWC" and out_fmt == "CHW":
        if arr.ndim == 3:
            return np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 4:
            return np.transpose(arr, (0, 3, 1, 2))
        elif arr.ndim == 5:
            return np.transpose(arr, (0, 1, 4, 2, 3))
    elif in_fmt == "CHW" and out_fmt == "HWC":
        if arr.ndim == 3:
            return np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 4:
            return np.transpose(arr, (0, 2, 3, 1))
        elif arr.ndim == 5:
            return np.transpose(arr, (0, 1, 2, 3, 4))

    raise ValueError(f"Unsupported transpose for shape {arr.shape} ({in_fmt}→{out_fmt})")


class ImageDataset(data.Dataset):
    def __init__(self, npz_path, in_format="HWC", out_format="CHW",
                 only_first=False, only_original=False):
        """
        Dataset over flat .npz arrays:
            img_o: (N, T, H, W, C)
            img_p: (N, T, H, W, C)
        Returns individual images.
        """
        data = np.load(npz_path)
        self.img_o = data["img_o"]
        self.img_p = data["img_p"]
        self.in_format, self.out_format = in_format, out_format
        self.only_first, self.only_original = only_first, only_original

        if self.img_o.max() > 1.0:
            self.img_o = self.img_o.astype(np.float32) / 255.0
            self.img_p = self.img_p.astype(np.float32) / 255.0

        N, T = self.img_o.shape[:2]
        self.idx2ep = [
            (n, t, pert)
            for n in range(N)
            for t in ([0] if only_first else range(T))
            for pert in ([False] if only_original else [False, True])
        ]

    def __len__(self):
        return len(self.idx2ep)

    def __getitem__(self, idx):
        n, t, is_pert = self.idx2ep[idx]
        img = self.img_p[n, t] if is_pert else self.img_o[n, t]
        img = transpose_array(img, self.in_format, self.out_format)
        return torch.from_numpy(img).float()
    

class PerturbedImageSequenceDataset(data.Dataset):
    def __init__(self, npz_path, in_format="HWC", out_format="CHW", T=None, only_original=False):
        """
        Loads full observation/perturbation sequences:
            img_o, img_p: (N, T, H, W, C)
            magnitudes, indices, properties: (N,)
        """
        data = np.load(npz_path, mmap_mode='r')
        assert T is None or T <= data["img_o"].shape[1], \
            "Requested T exceeds dataset length."
        self.in_format, self.out_format = in_format, out_format
        self.only_original = only_original
        self.img_o = data["img_o"][:, :T]

        if not only_original:
            self.img_p = data["img_p"][:, :T]
            self.mags = data["magnitudes"]
            self.inds = data["indices"]
            self.props = data["properties"]
        
    def __len__(self):
        return self.img_o.shape[0]

    def __getitem__(self, idx):
        orig = transpose_array(self.img_o[idx], self.in_format, self.out_format)
        if self.only_original:
            return torch.from_numpy(orig / 255.0).float()
        else:
            pert = transpose_array(self.img_p[idx], self.in_format, self.out_format)
            return (
                torch.from_numpy(orig / 255.0).float(),
                torch.from_numpy(pert / 255.0).float(),
                torch.tensor(self.mags[idx], dtype=torch.float32),
                torch.tensor(self.inds[idx], dtype=torch.int8),
                torch.tensor(self.props[idx], dtype=torch.int8),
            )


class ImageSequencePairDataset(data.Dataset):
    """
    Returns consecutive frame pairs (t, t+1) from the 'img_o' sequences.
    """
    def __init__(self, npz_path, in_format="HWC", out_format="CHW", only_first=True):
        data = np.load(npz_path)
        
        if only_first:
            self.img_o = data["img_o"][:, 0:2]
        else:
            self.img_o = data["img_o"]
            N, T = self.img_o.shape[:2]
            self.idx2pair = [(n, t) for n in range(N) for t in range(T - 1)]
        
        self.in_format, self.out_format = in_format, out_format
        self.change_type = True if self.img_o.max() > 1.0 else False
        self.only_first = only_first

    def __len__(self):
        return len(self.img_o)

    def __getitem__(self, idx):
        if self.only_first:  
            pair = self.img_o[idx]
        else:
            n, t = self.idx2pair[idx]
            pair = self.img_o[n, t:t+2]

        if self.change_type:
            pair = pair.astype(np.float32) / 255.0

        pair = transpose_array(pair, self.in_format, self.out_format)
        return torch.from_numpy(pair).float()


class SlotDataset(data.Dataset):
    """
    Loads slots from orig_seq and pert_seq (shape [B, T, O, S]) in a HDF5 file.
    Flattens all slots across time and objects, returning one randomly chosen slot
    (either original or perturbed) per __getitem__ call.
    """
    def __init__(self, hdf5_file, feature_mean=None, feature_std=None, normalize=True):
        self.hdf5_file = hdf5_file
        self.normalize = normalize
        self.slots = self._load_slots_data()

        if normalize:
            if feature_mean is None or feature_std is None:
                self.feature_mean = self.slots.mean(dim=0)  # Shape: [S]
                self.feature_std = self.slots.std(dim=0)    # Shape: [S]
            else:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
            self.slots = (self.slots - self.feature_mean) / self.feature_std


    def _load_slots_data(self):
        data = {
            'orig_seq': [],
            'pert_seq': []
        }

        with h5py.File(self.hdf5_file, 'r') as f:
            for key in f.keys():
                for k in data:
                    if k in key:
                        data[k].append(f[key][...])
                        break

        # Stack all data into single tensors
        orig_seq = torch.tensor(np.concatenate(data['orig_seq']), dtype=torch.float32)
        pert_seq = torch.tensor(np.concatenate(data['pert_seq']), dtype=torch.float32)
        sequences = torch.cat((orig_seq, pert_seq), dim=0)
        # Flatten the sequences across time and objects
        slots = sequences.reshape(-1, orig_seq.shape[-1])  # Shape: [B*T*O, S]
        return slots
    
    def _compute_mean_std(self):
        all_data = torch.tensor(np.concatenate([self.original, self.perturbed], axis=0), dtype=torch.float32)
        flat = all_data.reshape(-1, all_data.shape[-1])  # Shape: [B*T*O, E]
        mean = flat.mean(dim=0)  # Shape: [E]
        std = flat.std(dim=0)    # Shape: [E]
        return mean, std
        

    def __len__(self):
        return self.slots.shape[0]

    def __getitem__(self, idx):
        # Select a random slot regardless of whether it was original or perturbed
        return self.slots[idx]


class PerturbedSlotSequenceDataset(data.Dataset):
    """
    Dataset for loading perturbed slot sequences from a HDF5 file.
    Assumes the HDF5 file contains the following datasets:
        - 'orig_seq': Original slot sequences
        - 'pert_seq': Perturbed slot sequences
        - 'magnitude': Magnitude of perturbations
        - 'obj_index': Object indices
        - 'prop_index': Property indices

    The dataset can be normalized using the provided feature mean and standard deviation.
    If normalization is enabled but feature mean and standard deviation are not provided,
    the mean and standard deviation will be computed from the data.
    If only_first is True, only the first time step of each sequence is used.
    If prop_skip_codes is provided, samples with those property codes are filtered out.
    """
    def __init__(self, hdf5_file, feature_mean=None, feature_std=None, normalize=True, timesteps=None, prop_skip_codes: List[int] = None):
        self.hdf5_file = hdf5_file
        self.normalize = normalize
        self.T = timesteps
        self.prop_skip_codes = prop_skip_codes

        # Load all data into memory
        self.original, self.perturbed, self.magnitude, self.obj_index, self.prop_index = self._load_slots_data()
        self.num_samples = len(self.original)

        if self.normalize:
            if feature_mean is None or feature_std is None:
                self.feature_mean, self.feature_std = self._compute_mean_std()
            else:
                self.feature_mean = feature_mean
                self.feature_std = feature_std

            self.original = self._normalize_tensor(self.original)
            self.perturbed = self._normalize_tensor(self.perturbed)


    def _load_slots_data(self):
            data = { 'orig_seq': [], 'pert_seq': [],
                    'magnitude': [], 'obj_index': [], 'prop_index': [] }

            with h5py.File(self.hdf5_file, 'r') as f:
                for key in f.keys():
                    for k in data:
                        if k in key:
                            data[k].append(f[key][...])
                            break

            # Stack everything
            orig_seq = np.concatenate(data['orig_seq'], axis=0)
            pert_seq = np.concatenate(data['pert_seq'], axis=0)
            magnitude = np.concatenate(data['magnitude'], axis=0)
            obj_index = np.concatenate(data['obj_index'], axis=0)
            prop_index = np.concatenate(data['prop_index'], axis=0)

            orig_seq = orig_seq[:, :self.T]   # (B, O, S, T)
            pert_seq = pert_seq[:, :self.T]   # (B, O, S, T)

            if self.T == 1:
                orig_seq = np.squeeze(orig_seq, axis=-1)  # (B, O, S)
                pert_seq = np.squeeze(pert_seq, axis=-1)  # (B, O, S)

            if self.prop_skip_codes is not None:
                mask = ~np.isin(prop_index, self.prop_skip_codes)
                orig_seq = orig_seq[mask]
                pert_seq = pert_seq[mask]
                magnitude = magnitude[mask]
                obj_index = obj_index[mask]
                prop_index = prop_index[mask]

            # Convert to tensors
            orig_seq = torch.tensor(orig_seq, dtype=torch.float32)
            pert_seq = torch.tensor(pert_seq, dtype=torch.float32)
            magnitude = torch.tensor(magnitude, dtype=torch.float32)
            obj_index = torch.tensor(obj_index, dtype=torch.int64)
            prop_index = torch.tensor(prop_index, dtype=torch.int64)

            return orig_seq, pert_seq, magnitude, obj_index, prop_index

    def _compute_mean_std(self):
        with torch.no_grad():
            all_data = torch.tensor(np.concatenate([self.original, self.perturbed], axis=0), dtype=torch.float32)
            flat = all_data.reshape(-1, all_data.shape[-1])
            mean = flat.mean(dim=0)  # Shape: [E]
            std = flat.std(dim=0)    # Shape: [E]
            return mean, std

    def _normalize_tensor(self, x):
        self.feature_mean = self.feature_mean.to(x.device)
        self.feature_std = self.feature_std.to(x.device)
        return (x - self.feature_mean) / self.feature_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.original[idx],
                self.perturbed[idx],
                self.magnitude[idx],
                self.obj_index[idx],
                self.prop_index[idx])