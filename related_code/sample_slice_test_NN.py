'''
Mar 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from hyperparameters import Hyperparameters as hp
from data_load import *
from phantom_train_NN import Net
from tqdm import tqdm
from pdb import set_trace as bp


if __name__ == '__main__':
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # Data
    print("Loading data...")
    # df = pd.read_pickle('./data/sample_slice_LS_BP.pkl')
    df = pd.read_pickle('./data/sample_slice_LS_BP_3T.pkl')
    X = df.loc[:, ['b0', 'b10', 'b20', 'b60', 'b150', 'b300', 'b500', 'b1000']].values.astype('float32')
    # normalize dw signal
    S0 = X[:,0].copy()
    for i in range(len(hp.b_values)):
      X[:,i] = X[:,i]/S0
    X = X[:,1:]
    print("Done")

    # Network
    b_values_no0 = torch.FloatTensor(hp.b_values[1:])
    net = Net(b_values_no0).to(device)

    # Restore variables from disk
    # net.load_state_dict(torch.load(hp.logdir + 'sample_slice/final_model.pt', map_location=device))
    net.load_state_dict(torch.load(hp.logdir + 'sample_slice/final_model_3T.pt', map_location=device))

    # evaluate on test data
    net.eval()
    with torch.no_grad():
      X_pred, Dp_pred, Dt_pred, Fp_pred = net(torch.from_numpy(X.astype(np.float32)))

    df['Dp_NN'] = Dp_pred.numpy()
    df['Dt_NN'] = Dt_pred.numpy()
    df['Fp_NN'] = Fp_pred.numpy()

    # df.to_pickle('./data/sample_slice_LS_BP_NN.pkl')
    # df.to_csv('./data/sample_slice_LS_BP_NN.csv', index=False)
    df.to_pickle('./data/sample_slice_LS_BP_NN_3T.pkl')
    df.to_csv('./data/sample_slice_LS_BP_NN_3T.csv', index=False)


