'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import numpy as np
import scipy.io as sio
from skimage import io
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import pandas as pd
from pdb import set_trace as bp

if __name__ == '__main__':

  num_b_values = 8
  
  # # 1.5T
  # subject = 'a'
  # min_S0 = 10 # if less don't fit IVIM model  
  # z = 16
  # min_x = 20
  # max_x = 235
  # min_y = 70
  # max_y = 180
  
  # 3T
  subject = 'b'
  min_S0 = 50 # if less don't fit IVIM model  
  z = 8
  min_x = 25
  max_x = 235
  min_y = 65
  max_y = 190
  
  
  df = pd.DataFrame(columns=['subject', 'x', 'y', 'z', 'b0', 'b10', 'b20', 'b60', 'b150', 'b300', 'b500', 'b1000'])
  
  # image_path = 'H:/projects/IVIM/vendor_study/data/dwi_images/Philips_1_5T_' + subject + '_DWI_FB.tif'
  image_path = 'H:/projects/IVIM/vendor_study/data/dwi_images/Philips_3T_' + subject + '_DWI_FB.tif'
  image = io.imread(image_path)
  sz, sy, sx = image.shape
  z_one_b = sz//num_b_values
  
  for y in range(min_y, max_y):
    print(y)
    for x in range(min_x, max_x):
      if image[z, y, x] > min_S0:
        b_vals = image[z::z_one_b, y, x].tolist()
        record = [subject, x, y, z]
        for b_val in b_vals:
          record.append(b_val)
        df.loc[len(df)] = record

  # df.to_pickle('./data/sample_slice.pkl')
  # df.to_csv('./data/sample_slice.csv', index=False)
  df.to_pickle('./data/sample_slice_3T.pkl')
  df.to_csv('./data/sample_slice_3T.csv', index=False)  
  
  bp()
  
  plt.hist(image[z, min_y:max_y, min_x:max_x].ravel(), bins=256, fc='k', ec='k')
  plt.show()
  
  plt.imshow(image[z, min_y:max_y, min_x:max_x], cmap='gray', clim=(0, 600))
  plt.show()

  
