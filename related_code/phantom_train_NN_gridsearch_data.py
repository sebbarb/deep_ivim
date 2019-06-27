'''
Mar 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hyperparameters import Hyperparameters as hp
from data_load import *
from tqdm import tqdm
from pdb import set_trace as bp

class Net(nn.Module):
  def __init__(self, b_values_no0):
    super(Net, self).__init__()

    self.b_values_no0 = b_values_no0
    self.fc_layers = nn.ModuleList()
    for i in range(hp.num_blocks):
      self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
    self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 3))

  def forward(self, X):
    params = torch.abs(self.encoder(X)) # Dp, Dt, Fp
    Dp = params[:, 0].unsqueeze(1)
    Dt = params[:, 1].unsqueeze(1)
    Fp = params[:, 2].unsqueeze(1)

    X = Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt)
  
    return X, Dp, Dt, Fp


if __name__ == '__main__':
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # Store results
    n_runs = 10
    train_size_vec = np.array([1e3, 1e4, 1e5, 1e6]).astype(int)
    best_loss = np.zeros((n_runs, len(train_size_vec)))

    for run in range(n_runs):
      for size_id in range(len(train_size_vec)):
        train_size = train_size_vec[size_id]
      
        # Training data
        print("Loading training data...")
        X, Dp, Dt, Fp = load_data(hp.data_file)
        X_all = X
        train_idx = np.random.choice(X.shape[0], train_size, replace=False)
        X_train, Dp_train, Dt_train, Fp_train = X[train_idx, :], Dp[train_idx, :], Dt[train_idx, :], Fp[train_idx, :]
        trainloader, num_batches = get_trainloader(X_train, Dp_train, Dt_train, Fp_train)
        print("Done")

        # Network
        b_values_no0 = torch.FloatTensor(hp.b_values[1:]).to(device)
        net = Net(b_values_no0).to(device)

        # Loss function and optimizer
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(net.parameters(), lr = 0.001)  

        # Best loss
        best = 1e16
        num_bad_epochs = 0
        
        # Train
        for epoch in range(hp.num_epochs): 
          print("-----------------------------------------------------------------")
          print("Run: {}; Train size: {}; Epoch: {}; Bad epochs: {}".format(run, train_size, epoch, num_bad_epochs))
          net.train()
          running_loss = 0.
          
          for i, (X, Dp, Dt, Fp) in enumerate(tqdm(trainloader), 0):
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
              # torch.save(final_model, hp.logdir + 'phantom/final_model.pt')
              X_pred, Dp_pred, Dt_pred, Fp_pred = net(torch.from_numpy(X_all.astype(np.float32)))
              loss_all = criterion(X_pred, torch.from_numpy(X_all.astype(np.float32)))
              best_loss[run, size_id] = loss_all
              np.savez('./data/gridsearch_data.npz', best_loss=best_loss)
              print("Done, best loss: {}, overall loss: {}".format(best, loss_all))
              break
    print("Done")

