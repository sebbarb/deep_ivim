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
    
    # Training data
    print("Loading training data...")
    # df = pd.read_pickle('./data/sample_slice.pkl')
    df = pd.read_pickle('./data/sample_slice_3T.pkl')
    X = df.loc[:, ['b0', 'b10', 'b20', 'b60', 'b150', 'b300', 'b500', 'b1000']].values.astype('float32')
    # normalize dw signal
    S0 = X[:,0].copy()
    for i in range(len(hp.b_values)):
      X[:,i] = X[:,i]/S0
    X = X[:,1:]
    trainloader, num_batches = get_trainloader(X)
    print("Done")

    # Network
    b_values_no0 = torch.FloatTensor(hp.b_values[1:])
    net = Net(b_values_no0).to(device)

    # Restore variables from disk
    net.load_state_dict(torch.load(hp.logdir + 'phantom/final_model_Dp02.pt', map_location=device))

    # Loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr = 0.001)  

    # Best loss
    best = 1e16
    num_bad_epochs = 0
    
    # Train
    for epoch in range(hp.num_epochs): 
      print("-----------------------------------------------------------------")
      print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
      net.train()
      running_loss = 0.
      
      for i, X in enumerate(tqdm(trainloader), 0):
        # move to GPU if available
        X = X.to(device)
      
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        X_pred, Dp_pred, Dt_pred, Fp_pred = net(X)
        loss = criterion(X_pred, X)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      
      print("Loss: {}".format(running_loss))
      # early stopping
      if running_loss < best:
        print("############### Saving good model ###############################")
        final_model = net.state_dict()
        best = running_loss
        num_bad_epochs = 0
      else:
        num_bad_epochs = num_bad_epochs + 1
        if num_bad_epochs == hp.patience:
          # torch.save(final_model, hp.logdir + 'sample_slice/final_model.pt')
          torch.save(final_model, hp.logdir + 'sample_slice/final_model_3T.pt')
          print("Done, best loss: {}".format(best))
          break
    print("Done")


  
  
    

