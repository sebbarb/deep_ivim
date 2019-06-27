'''
Mar 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

class Hyperparameters:
  '''Hyperparameters'''
  
  # data files and random states
  data_file = './data/train_Dp01.npz' # train data
  rs_1 = 123 # random seed for data_gen.py
  rs_2 = 456 # random seed for data_gen.py
  
  # data_file = './data/test_Dp01.npz' # test data
  # rs_1 = 123+10 # random seed for data_gen.py
  # rs_2 = 456+10 # random seed for data_gen.py
  
  # data parameters
  # num_samples = 1000000
  num_samples = 10000
  b_values = [0,10,20,60,150,300,500,1000]
  S0 = 1500
  Dp_min = 0.01
  Dp_max = 0.1
  # Dp_max = 0.2
  Dt_min = 0.0005
  Dt_max = 0.002
  Fp_min = 0.1
  Fp_max = 0.4
  # noise parameters
  add_noise = True
  noise_sd_min = 0
  noise_sd_max = 165
  # noise_sd_min = 150
  # noise_sd_max = noise_sd_min

  # training
  batch_size = 128 # alias = N
  logdir = './logdir/' # log directory
  num_blocks = 3 # number of feedforward hidden layers with dropout
  num_epochs = 1000
  patience = 10 # early stopping
    
    
    
    
