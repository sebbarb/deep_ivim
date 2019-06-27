'''
Mar 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

from hyperparameters import Hyperparameters as hp
import numpy as np
import torch
import torch.utils.data as utils
from pdb import set_trace as bp

def load_data(data_file):
  npzfile = np.load(data_file)
  X, Dp, Dt, Fp = npzfile["X_dw"], npzfile["Dp"], npzfile["Dt"], npzfile["Fp"]
  return X, Dp, Dt, Fp


def get_trainloader(X, Dp=None, Dt=None, Fp=None):
  # Compute total batch count
  num_batches = len(X) // hp.batch_size
  
  # Create dataset
  if (Dp is None) and (Dt is None) and (Fp is None):
    dataset = torch.from_numpy(X.astype(np.float32))
  else:
    dataset = utils.TensorDataset(torch.from_numpy(X.astype(np.float32)), 
                                  torch.from_numpy(Dp.astype(np.float32)),
                                  torch.from_numpy(Dt.astype(np.float32)),
                                  torch.from_numpy(Fp.astype(np.float32)))
          
  # Create batch queues
  trainloader = utils.DataLoader(dataset,
                                 batch_size = hp.batch_size, 
                                 shuffle = True,
                                 num_workers = 2,
                                 drop_last = True)
  
  return trainloader, num_batches

