'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp

if __name__ == '__main__':

  fg, ax = plt.subplots(3, 1, sharex='all')

  data = np.load('./data/rmse_LS_BP_NN.npz')

  Dp_rmse_LS, Dt_rmse_LS, Fp_rmse_LS = data['Dp_rmse_LS'], data['Dt_rmse_LS'], data['Fp_rmse_LS']
  Dp_rmse_BP, Dt_rmse_BP, Fp_rmse_BP = data['Dp_rmse_BP'], data['Dt_rmse_BP'], data['Fp_rmse_BP']
  Dp_rmse_NN, Dt_rmse_NN, Fp_rmse_NN = data['Dp_rmse_NN'], data['Dt_rmse_NN'], data['Fp_rmse_NN']
  
  LS = ax[2].plot(range(10), Dp_rmse_LS, label='LS')
  BP = ax[2].plot(range(10), Dp_rmse_BP, label='BP')
  NN = ax[2].plot(range(10), Dp_rmse_NN, label='DNN')
  ax[2].legend()
  ax[2].set(ylabel=r'$D_p$ [mm$^2$/sec]')
  ax[2].set(xlabel='SNR')
  ax[2].set_xticks(range(10))
  ax[2].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])
  
  ax[0].plot(range(10), Dt_rmse_LS, label='LS')
  ax[0].plot(range(10), Dt_rmse_BP, label='BP')
  ax[0].plot(range(10), Dt_rmse_NN, label='DNN')
  ax[0].legend()
  ax[0].set(ylabel=r'$D_t$ [mm$^2$/sec]')
  ax[0].set_xticks(range(10))
  ax[0].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])
  ax[0].set_title('Root-Mean-Square Error')

  ax[1].plot(range(10), Fp_rmse_LS*100, label='LS')
  ax[1].plot(range(10), Fp_rmse_BP*100, label='BP')
  ax[1].plot(range(10), Fp_rmse_NN*100, label='DNN')
  ax[1].legend()
  ax[1].set(ylabel=r'$F_p$ [%]')
  ax[1].set_xticks(range(10))
  ax[1].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])

  fg.tight_layout()
  # plt.show()

  fg.savefig('./fig/phantom_rmse.pdf', bbox_inches='tight')
  fg.savefig('./fig/phantom_rmse.tif', dpi=fg.dpi, bbox_inches='tight')
  
  

  
