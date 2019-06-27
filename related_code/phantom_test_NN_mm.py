'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

from hyperparameters import Hyperparameters as hp
import numpy as np
import torch
from phantom_train_NN import Net
from data_load import *
import matplotlib.pyplot as plt
from time import time
from pdb import set_trace as bp

if __name__ == '__main__':
  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True
  
  # Network
  b_values_no0 = torch.FloatTensor(hp.b_values[1:])
  net = Net(b_values_no0).to(device)
  
  Dp_pred = np.zeros((hp.num_samples, 10))
  Dt_pred = np.zeros((hp.num_samples, 10))
  Fp_pred = np.zeros((hp.num_samples, 10))

  Dp_error = np.zeros((hp.num_samples, 10))
  Dt_error = np.zeros((hp.num_samples, 10))
  Fp_error = np.zeros((hp.num_samples, 10))

  time_start = time()
  for noise_sd in range(15, 165, 15):
    print(noise_sd)
    # Load test data
    print("Loading test data...")
    X, Dp_truth, Dt_truth, Fp_truth = load_data('./data/test_noise' + str(noise_sd) + '.npz')
    print("Done")

    # Restore variables from disk
    net.load_state_dict(torch.load(hp.logdir + 'phantom/final_model_' + str(noise_sd) + '.pt', map_location=device))
        
    # evaluate on test data
    net.eval()
    with torch.no_grad():
      X_pred, Dp, Dt, Fp = net(torch.from_numpy(X.astype(np.float32)))
      
    id = noise_sd//15-1
    Dp_pred[:,id] = np.ravel(Dp.numpy())
    Dt_pred[:,id] = np.ravel(Dt.numpy())
    Fp_pred[:,id] = np.ravel(Fp.numpy())
    Dp_error[:,id] = np.ravel(Dp.numpy()-Dp_truth)
    Dt_error[:,id] = np.ravel(Dt.numpy()-Dt_truth)
    Fp_error[:,id] = np.ravel(Fp.numpy()-Fp_truth)    

  time_end = time()
  time = time_end-time_start

  data_file = './data/phantom_pred_NN_new'
  np.savez(data_file, Dp_pred=Dp_pred, Dt_pred=Dt_pred, Fp_pred=Fp_pred, time=time)

  data_file = './data/phantom_errors_NN_new'
  np.savez(data_file, Dp_error=Dp_error, Dt_error=Dt_error, Fp_error=Fp_error, time=time)
  
  print('Time: {}'.format(time))
  
  plt.boxplot(Dp_error)
  plt.show()
  plt.boxplot(Dt_error)
  plt.show()
  plt.boxplot(Fp_error)
  plt.show()

