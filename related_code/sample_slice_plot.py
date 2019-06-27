'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as patches
from pdb import set_trace as bp

if __name__ == '__main__':

  # # 1.5T
  # min_x = 20
  # max_x = 235
  # min_y = 70
  # max_y = 180

  # 3T
  min_x = 25
  max_x = 235
  min_y = 65
  max_y = 190

  sx = 256
  sy = 256
  
  S0 = np.zeros((sy, sx))
  Dp_LS = np.zeros((sy, sx))
  Dt_LS = np.zeros((sy, sx))
  Fp_LS = np.zeros((sy, sx))
  Dp_BP = np.zeros((sy, sx))
  Dt_BP = np.zeros((sy, sx))
  Fp_BP = np.zeros((sy, sx))
  Dp_NN = np.zeros((sy, sx))
  Dt_NN = np.zeros((sy, sx))
  Fp_NN = np.zeros((sy, sx))
  
  # df = pd.read_pickle('./data/sample_slice_LS_BP_NN.pkl')
  df = pd.read_pickle('./data/sample_slice_LS_BP_NN_3T.pkl')
  for index, row in df.iterrows():
    x = row['x']
    y = row['y']
    
    S0[y, x] = row['b0']
    
    Dp_LS[y, x] = row['Dp_LS']
    Dt_LS[y, x] = row['Dt_LS']
    Fp_LS[y, x] = row['Fp_LS']
    
    Dp_BP[y, x] = row['Dp_BP']
    Dt_BP[y, x] = row['Dt_BP']
    Fp_BP[y, x] = row['Fp_BP']

    Dp_NN[y, x] = row['Dp_NN']
    Dt_NN[y, x] = row['Dt_NN']
    Fp_NN[y, x] = row['Fp_NN']
    
  S0 = S0[min_y:max_y, min_x:max_x]
  
  Dp_LS = Dp_LS[min_y:max_y, min_x:max_x]
  Dt_LS = Dt_LS[min_y:max_y, min_x:max_x]
  Fp_LS = Fp_LS[min_y:max_y, min_x:max_x]
  
  Dp_BP = Dp_BP[min_y:max_y, min_x:max_x]
  Dt_BP = Dt_BP[min_y:max_y, min_x:max_x]
  Fp_BP = Fp_BP[min_y:max_y, min_x:max_x]

  Dp_NN = Dp_NN[min_y:max_y, min_x:max_x]
  Dt_NN = Dt_NN[min_y:max_y, min_x:max_x]
  Fp_NN = Fp_NN[min_y:max_y, min_x:max_x]

  fig = plt.figure()
  grid = ImageGrid(fig, 111,
                   nrows_ncols=(4,3),
                   direction='row',
                   axes_pad=0.2,
                   cbar_location='right',
                   cbar_mode='edge',
                   cbar_size='2%',
                   cbar_pad=0.15)

  # grid[0].imshow(S0, cmap='gray', clim=(0, 600))  
  grid[0].imshow(S0, cmap='gray', clim=(0, 1200))  
  
  # grid[3].imshow(Dt_LS, cmap='gray', clim=(0, 0.0025))
  # grid[4].imshow(Dt_BP, cmap='gray', clim=(0, 0.0025))
  # cp_Dt = grid[5].imshow(Dt_NN, cmap='gray', clim=(0, 0.0025))
  grid[3].imshow(Dt_LS, cmap='gray', clim=(0, 0.003))
  grid[4].imshow(Dt_BP, cmap='gray', clim=(0, 0.003))
  cp_Dt = grid[5].imshow(Dt_NN, cmap='gray', clim=(0, 0.003))  
  
  grid[6].imshow(Fp_LS*100, cmap='gray', clim=(0, 70))
  grid[7].imshow(Fp_BP*100, cmap='gray', clim=(0, 70))
  cp_Fp = grid[8].imshow(Fp_NN*100, cmap='gray', clim=(0, 70))
  
  grid[9].imshow(Dp_LS, cmap='gray', clim=(0, 0.1))
  grid[10].imshow(Dp_BP, cmap='gray', clim=(0, 0.1))
  cp_Dp = grid[11].imshow(Dp_NN, cmap='gray', clim=(0, 0.1))  

  grid[9].set_xlabel('Least-Squares')
  grid[10].set_xlabel('Bayesian')
  grid[11].set_xlabel('Neural Network')

  grid[0].set_ylabel(r'b=0')
  grid[3].set_ylabel(r'$D_t$ [mm$^2$/sec]')
  grid[6].set_ylabel(r'$F_p$ [%]')
  grid[9].set_ylabel(r'$D_p$ [mm$^2$/sec]')

  grid[5].cax.colorbar(cp_Dt)
  grid[8].cax.colorbar(cp_Fp)
  grid[11].cax.colorbar(cp_Dp)

  for i, axis in enumerate(grid):
    axis.set_xticks([])
    axis.set_yticks([])
    if (i==1) or (i==2):
      axis.set_axis_off()

  plt.tight_layout()

  # fig.savefig('./fig/sample_slice.pdf', bbox_inches='tight')
  # fig.savefig('./fig/sample_slice.tif', dpi=fig.dpi, bbox_inches='tight')
  fig.savefig('./fig/sample_slice_3T.pdf', bbox_inches='tight')
  fig.savefig('./fig/sample_slice_3T.tif', dpi=fig.dpi, bbox_inches='tight')

  
