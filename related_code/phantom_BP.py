'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

from hyperparameters import Hyperparameters as hp
import numpy as np
from data_load import *
import matplotlib.pyplot as plt
from time import time
from fitting_functions import *
from scipy import stats
from tqdm import tqdm
from pdb import set_trace as bp

if __name__ == '__main__':
  # Empirical Prior from least squares fits 
  data = np.load('./data/phantom_pred_LS.npz')
  Dp0, Dt0, Fp0 = data['Dp_pred'].flatten(), data['Dt_pred'].flatten(), data['Fp_pred'].flatten()
  neg_log_prior = empirical_neg_log_prior(Dp0, Dt0, Fp0)

  b_values_no0 = np.array(hp.b_values[1:])

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
    
    id = noise_sd//15-1
    
    for i in tqdm(range(hp.num_samples)):
      Dp, Dt, Fp = fit_bayesian(b_values_no0, X[i,:], neg_log_prior)
      Dp_pred[i,id] = Dp
      Dt_pred[i,id] = Dt
      Fp_pred[i,id] = Fp      
      Dp_error[i,id] = np.ravel(Dp-Dp_truth[i])
      Dt_error[i,id] = np.ravel(Dt-Dt_truth[i])
      Fp_error[i,id] = np.ravel(Fp-Fp_truth[i])

  time_end = time()
  time = time_end-time_start

  data_file = './data/phantom_pred_BP'
  np.savez(data_file, Dp_pred=Dp_pred, Dt_pred=Dt_pred, Fp_pred=Fp_pred, time=time)

  data_file = './data/phantom_errors_BP'
  np.savez(data_file, Dp_error=Dp_error, Dt_error=Dt_error, Fp_error=Fp_error, time=time)
  
  print('Time: {}'.format(time))


