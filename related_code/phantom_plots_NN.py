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
from scipy.stats import uniform
from pdb import set_trace as bp

if __name__ == '__main__':
  # Load test data
  print("Loading test data...")
  X_Dp01, _, _, _ = load_data('./data/test_Dp01.npz')
  X_Dp02, _, _, _ = load_data('./data/test_Dp02.npz')
  print("Done")

  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True
  
  # Network
  b_values_no0 = torch.FloatTensor(hp.b_values[1:])
  net = Net(b_values_no0).to(device) 
  
  # Restore variables from disk
  net.load_state_dict(torch.load(hp.logdir + 'phantom/final_model_Dp01.pt', map_location=device))
        
  # evaluate on test data
  print("Evaluating test data 1...")
  net.eval()
  with torch.no_grad():
    X_pred_Dp01, Dp_Dp01, Dt_Dp01, Fp_Dp01 = net(torch.from_numpy(X_Dp01.astype(np.float32)))
  print("Done")
      
  # Restore variables from disk
  net.load_state_dict(torch.load(hp.logdir + 'phantom/final_model_Dp02.pt', map_location=device))
        
  # evaluate on test data
  print("Evaluating test data 2...")
  net.eval()
  with torch.no_grad():
    X_pred_Dp02, Dp_Dp02, Dt_Dp02, Fp_Dp02 = net(torch.from_numpy(X_Dp02.astype(np.float32)))
  print("Done")
  
  bp()

  # plot
  fg, ax = plt.subplots(2, 2)

  x = np.linspace(0, 0.003, 100)
  ax[0,0].set_xlim([min(x), max(x)])
  ax[0,0].hist(Dt_Dp01.numpy(), 100, density=True, facecolor='#029386')
  ax[0,0].plot(x, uniform.pdf(x, hp.Dt_min, hp.Dt_max-hp.Dt_min), c='black')
  ax[0,0].yaxis.set_visible(False)
  ax[0,0].set(xlabel=r'$D_t$ [mm$^2$/sec]')

  x = np.linspace(0, 70, 100)
  ax[0,1].set_xlim([min(x), max(x)])
  ax[0,1].hist(Fp_Dp01.numpy()*100, 100, density=True, facecolor='#ffb07c')
  ax[0,1].plot(x, uniform.pdf(x, hp.Fp_min*100, (hp.Fp_max-hp.Fp_min)*100), c='black')
  ax[0,1].yaxis.set_visible(False)
  ax[0,1].set(xlabel=r'$F_p$ [%]')

  x = np.linspace(0, 0.2, 100)
  ax[1,0].set_xlim([min(x), max(x)])
  ax[1,0].hist(Dp_Dp01.numpy(), 100, density=True, facecolor='#a2cffe')
  ax[1,0].plot(x, uniform.pdf(x, hp.Dp_min, hp.Dp_max-hp.Dp_min), c='black')
  ax[1,0].yaxis.set_visible(False)
  ax[1,0].set(xlabel=r'$D_p$ [mm$^2$/sec]')

  x = np.linspace(0, 0.4, 100)
  ax[1,1].set_xlim([min(x), max(x)])
  ax[1,1].hist(Dp_Dp02.numpy(), 100, density=True, facecolor='#a2cffe')
  ax[1,1].plot(x, uniform.pdf(x, hp.Dp_min, 0.2-hp.Dp_min), c='black')
  ax[1,1].yaxis.set_visible(False)
  ax[1,1].set(xlabel=r'$D_p$ [mm$^2$/sec]')

  plt.show()

  # fg.savefig('./fig/phantom_plots_NN.pdf', bbox_inches='tight')
  # fg.savefig('./fig/phantom_plots_NN.tif', dpi=fg.dpi, bbox_inches='tight')

  
