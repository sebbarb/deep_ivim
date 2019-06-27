'''
Feb 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

from hyperparameters import Hyperparameters as hp
import numpy as np
import pandas as pd
from data_load import *
from fitting_functions import *
from tqdm import tqdm
from pdb import set_trace as bp

if __name__ == '__main__':
  # df = pd.read_pickle('./data/abdominal_data_LS.pkl')
  df = pd.read_pickle('./data/abdominal_data_LS_3T.pkl')
  X = df.loc[:, ['b0', 'b10', 'b20', 'b60', 'b150', 'b300', 'b500', 'b1000']].values.astype('float32')
  # normalize dw signal
  S0 = X[:,0].copy()
  for i in range(len(hp.b_values)):
    X[:,i] = X[:,i]/S0
  X = X[:,1:]
  
  # Empirical Prior from least squares fits 
  Dp0, Dt0, Fp0 = df['Dp_LS'], df['Dt_LS'], df['Fp_LS']
  neg_log_prior = empirical_neg_log_prior(Dp0, Dt0, Fp0)

  b_values_no0 = np.array(hp.b_values[1:])

  Dp_pred = np.zeros(len(X))
  Dt_pred = np.zeros(len(X))
  Fp_pred = np.zeros(len(X))

  for i in tqdm(range(len(X))):
    Dp, Dt, Fp = fit_bayesian(b_values_no0, X[i,:], neg_log_prior)
    Dp_pred[i] = Dp
    Dt_pred[i] = Dt
    Fp_pred[i] = Fp
    
  df['Dp_BP'] = Dp_pred
  df['Dt_BP'] = Dt_pred
  df['Fp_BP'] = Fp_pred

  # df.to_pickle('./data/abdominal_data_LS_BP.pkl')
  # df.to_csv('./data/abdominal_data_LS_BP.csv', index=False)
  df.to_pickle('./data/abdominal_data_LS_BP_3T.pkl')
  df.to_csv('./data/abdominal_data_LS_BP_3T.csv', index=False)
