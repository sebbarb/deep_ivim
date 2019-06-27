'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp

if __name__ == '__main__':

  fg, ax = plt.subplots(3, 3, sharex='all', sharey='row')

  data = np.load('./data/phantom_errors_LS.npz')

  Dp = data['Dp_error']
  Dp_mask = ~np.isnan(Dp)
  Dp_filtered = [d[m] for d, m in zip(Dp.T, Dp_mask.T)]
  ax[2,0].boxplot(Dp_filtered, showfliers=False)
  ax[2,0].set(ylabel=r'($D_p$ fit)-($D_p$ true) [mm$^2$/sec]')
  ax[2,0].axhline(linestyle='--')
  ax[2,0].set(xlabel='SNR')
  ax[2,0].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])
  
  Dt = data['Dt_error']
  Dt_mask = ~np.isnan(Dt)
  Dt_filtered = [d[m] for d, m in zip(Dt.T, Dt_mask.T)]
  ax[0,0].boxplot(Dt_filtered, showfliers=False)
  ax[0,0].set(ylabel=r'($D_t$ fit)-($D_t$ true) [mm$^2$/sec]')
  ax[0,0].set_title('Least-Squares')
  ax[0,0].axhline(linestyle='--')
  ax[0,0].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])  

  Fp = data['Fp_error']*100
  Fp_mask = ~np.isnan(Fp)
  Fp_filtered = [d[m] for d, m in zip(Fp.T, Fp_mask.T)]
  ax[1,0].boxplot(Fp_filtered, showfliers=False)
  ax[1,0].set(ylabel=r'($F_p$ fit)-($F_p$ true) [%]')
  ax[1,0].axhline(linestyle='--')
  ax[1,0].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])
  

  data = np.load('./data/phantom_errors_BP.npz')

  Dp = data['Dp_error']
  Dp_mask = ~np.isnan(Dp)
  Dp_filtered = [d[m] for d, m in zip(Dp.T, Dp_mask.T)]
  ax[2,1].boxplot(Dp_filtered, showfliers=False)
  ax[2,1].axhline(linestyle='--')
  ax[2,1].set(xlabel='SNR')
  ax[2,1].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])
  
  Dt = data['Dt_error']
  Dt_mask = ~np.isnan(Dt)
  Dt_filtered = [d[m] for d, m in zip(Dt.T, Dt_mask.T)]
  ax[0,1].boxplot(Dt_filtered, showfliers=False)
  ax[0,1].set_title('Bayesian')
  ax[0,1].axhline(linestyle='--')
  ax[0,1].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])

  Fp = data['Fp_error']*100
  Fp_mask = ~np.isnan(Fp)
  Fp_filtered = [d[m] for d, m in zip(Fp.T, Fp_mask.T)]
  ax[1,1].boxplot(Fp_filtered, showfliers=False)
  ax[1,1].axhline(linestyle='--')
  ax[1,1].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])


  data = np.load('./data/phantom_errors_NN.npz')

  Dp = data['Dp_error']
  Dp_mask = ~np.isnan(Dp)
  Dp_filtered = [d[m] for d, m in zip(Dp.T, Dp_mask.T)]
  ax[2,2].boxplot(Dp_filtered, showfliers=False)
  ax[2,2].axhline(linestyle='--')
  ax[2,2].set(xlabel='SNR')
  ax[2,2].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])
  
  Dt = data['Dt_error']
  Dt_mask = ~np.isnan(Dt)
  Dt_filtered = [d[m] for d, m in zip(Dt.T, Dt_mask.T)]
  ax[0,2].boxplot(Dt_filtered, showfliers=False)
  ax[0,2].set_title('DNN')
  ax[0,2].axhline(linestyle='--')
  ax[0,2].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])

  Fp = data['Fp_error']*100
  Fp_mask = ~np.isnan(Fp)
  Fp_filtered = [d[m] for d, m in zip(Fp.T, Fp_mask.T)]
  ax[1,2].boxplot(Fp_filtered, showfliers=False)
  ax[1,2].axhline(linestyle='--')
  ax[1,2].set_xticklabels([100, 50, 33, 25, 20, 17, 14, 13, 11, 10])

  fg.tight_layout()
  plt.show()

  # fg.savefig('./fig/phantom_boxplots.pdf', bbox_inches='tight')
  # fg.savefig('./fig/phantom_boxplots.tif', dpi=fg.dpi, bbox_inches='tight')
  
  

  
