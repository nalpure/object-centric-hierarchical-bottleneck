import h5py
import imageio
import matplotlib
import numpy
import scipy
import torch

try:
    import tomli
except ImportError:
    import tomllib as tomli

print("Core imports OK")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())