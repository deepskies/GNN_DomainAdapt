""" 
Constants used in the project.

"""

import random

import numpy as np
import torch

# --- RANDOM SEEDS --- #
torch.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)
    
# --- GLOBAL CONSTANTS --- #

# DATA CONSTANTS
HRED = 0.7                              # Reduced Hubble constant
BOX_SIZE = 25e3                         # Box size in comoving kpc/h
VALID_SIZE, TEST_SIZE = 0.15, 0.15      # Validation and test size
BATCH_SIZE = 32                         # Batch size    
Nstar_th = 20                           # Minimum number of stellar particles required to consider a galaxy

# Use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

