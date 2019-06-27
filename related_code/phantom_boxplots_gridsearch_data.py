'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp

if __name__ == '__main__':

  best_loss = np.load('./data/gridsearch_data.npz')['best_loss']

  print(np.transpose(np.round(np.quantile(best_loss*1e6/128, [0.25, 0.50, 0.75], axis=0)*1000)/1000))
