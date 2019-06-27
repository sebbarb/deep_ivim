'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import numpy as np
from data_load import *
from pdb import set_trace as bp

def rmse(predictions, targets):
  return np.sqrt( np.mean((predictions - targets) ** 2) )

if __name__ == '__main__':

  # fifth column corresponds to noise sigma of 75 (SNR of 20)

  data = np.load('./data/phantom_pred_LS.npz')
  Dp_LS = data['Dp_pred']
  Dt_LS = data['Dt_pred']
  Fp_LS = data['Fp_pred']

  data = np.load('./data/phantom_pred_BP.npz')
  Dp_BP = data['Dp_pred']
  Dt_BP = data['Dt_pred']
  Fp_BP = data['Fp_pred']

  data = np.load('./data/phantom_pred_NN.npz')
  Dp_NN = data['Dp_pred']
  Dt_NN = data['Dt_pred']
  Fp_NN = data['Fp_pred']
  
  Dp_rmse_LS, Dt_rmse_LS, Fp_rmse_LS = np.zeros(10), np.zeros(10), np.zeros(10)
  Dp_rmse_BP, Dt_rmse_BP, Fp_rmse_BP = np.zeros(10), np.zeros(10), np.zeros(10)
  Dp_rmse_NN, Dt_rmse_NN, Fp_rmse_NN = np.zeros(10), np.zeros(10), np.zeros(10)
  
  for noise_sd in range(15, 165, 15):
    print(noise_sd)
    # Load test data
    print("Loading test data...")
    X, Dp_truth, Dt_truth, Fp_truth = load_data('./data/test_noise' + str(noise_sd) + '.npz')
    Dp_truth, Dt_truth, Fp_truth = Dp_truth.T, Dt_truth.T, Fp_truth.T
    print("Done")
    
    id = noise_sd//15-1
    Dp_rmse_LS[id], Dt_rmse_LS[id], Fp_rmse_LS[id] = rmse(Dp_LS[:, id], Dp_truth), rmse(Dt_LS[:, id], Dt_truth), rmse(Fp_LS[:, id], Fp_truth)
    Dp_rmse_BP[id], Dt_rmse_BP[id], Fp_rmse_BP[id] = rmse(Dp_BP[:, id], Dp_truth), rmse(Dt_BP[:, id], Dt_truth), rmse(Fp_BP[:, id], Fp_truth)
    Dp_rmse_NN[id], Dt_rmse_NN[id], Fp_rmse_NN[id] = rmse(Dp_NN[:, id], Dp_truth), rmse(Dt_NN[:, id], Dt_truth), rmse(Fp_NN[:, id], Fp_truth)
    
  data_file = './data/rmse_LS_BP_NN'
  np.savez(data_file, Dp_rmse_LS=Dp_rmse_LS, Dt_rmse_LS=Dt_rmse_LS, Fp_rmse_LS=Fp_rmse_LS,
                      Dp_rmse_BP=Dp_rmse_BP, Dt_rmse_BP=Dt_rmse_BP, Fp_rmse_BP=Fp_rmse_BP,
                      Dp_rmse_NN=Dp_rmse_NN, Dt_rmse_NN=Dt_rmse_NN, Fp_rmse_NN=Fp_rmse_NN)
