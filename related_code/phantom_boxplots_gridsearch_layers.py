'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp

if __name__ == '__main__':

  fg, ax = plt.subplots()

  best_loss = np.load('./data/gridsearch_layers.npz')['best_loss']

  ax.boxplot(best_loss)
  ax.set(ylabel='MSE Loss')
  ax.set(xlabel='Number of Hidden Layers')
  ax.set_xticklabels(range(10))
  
  fg.tight_layout()
  # plt.show()

  fg.savefig('./fig/gridsearch_boxplots.pdf', bbox_inches='tight')
  fg.savefig('./fig/gridsearch_boxplots.tif', dpi=fg.dpi, bbox_inches='tight')
  
  

  
