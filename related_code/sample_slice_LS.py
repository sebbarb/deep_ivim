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
  # df = pd.read_pickle('./data/sample_slice.pkl')
  df = pd.read_pickle('./data/sample_slice_3T.pkl')
  X = df.loc[:, ['b0', 'b10', 'b20', 'b60', 'b150', 'b300', 'b500', 'b1000']].values.astype('float32')
  # normalize dw signal
  S0 = X[:,0].copy()
  for i in range(len(hp.b_values)):
    X[:,i] = X[:,i]/S0
  X = X[:,1:]

  b_values_no0 = np.array(hp.b_values[1:])

  Dp_pred = np.zeros(len(X))
  Dt_pred = np.zeros(len(X))
  Fp_pred = np.zeros(len(X))

  for i in tqdm(range(len(X))):
    Dp, Dt, Fp = fit_least_squares(b_values_no0, X[i,:])
    Dp_pred[i] = Dp
    Dt_pred[i] = Dt
    Fp_pred[i] = Fp
    
  df['Dp_LS'] = Dp_pred
  df['Dt_LS'] = Dt_pred
  df['Fp_LS'] = Fp_pred

  # df.to_pickle('./data/sample_slice_LS.pkl')
  # df.to_csv('./data/sample_slice_LS.csv', index=False)
  df.to_pickle('./data/sample_slice_LS_3T.pkl')
  df.to_csv('./data/sample_slice_LS_3T.csv', index=False)

