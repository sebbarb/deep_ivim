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

def region_map(name):
  region_dictionary = {'LS_III_left_pv':            'left_liver_lobe',
                       'LS_IV_adj_GB':              'left_liver_lobe',
                       'LS_III_left_pv_anterior':   'left_liver_lobe',
                       'LS_VI_lowest':              'right_liver_lobe',
                       'LS_V_adj_GB':               'right_liver_lobe',
                       'LS_VI_adj_Kidney':          'right_liver_lobe',
                       'Pancreas_caput':            'pancreas',
                       'Pancreas_corpus':           'pancreas',
                       'Pancreas_cauda':            'pancreas',
                       'Spleen':                    'spleen',
                       'KidneyR_UpperPole_cortex':  'renal_cortex',
                       'KidneyR_PI_cortex':         'renal_cortex',
                       'KidneyR_LowerPole_cortex':  'renal_cortex',
                       'KidneyL_UpperPole_cortex':  'renal_cortex',
                       'KidneyL_PI_cortex':         'renal_cortex',
                       'KidneyL_LowerPole_cortex':  'renal_cortex',
                       'KidneyR_UpperPole_medulla': 'renal_medulla',
                       'KidneyR_PI_medulla':        'renal_medulla',
                       'KidneyR_LowerPole_medulla': 'renal_medulla',
                       'KidneyL_UpperPole_medulla': 'renal_medulla',
                       'KidneyL_PI_medulla':        'renal_medulla',
                       'KidneyL_LowerPole_medulla': 'renal_medulla'}
  for key, value in region_dictionary.items():
    if key.lower() in name.lower():
      return value
  return None

if __name__ == '__main__':

  num_b_values = 8
  subject_list = ['a',
                  'b',
                  'c',
                  'd',
                  'e',
                  'f',
                  'g',
                  'h',
                  'i',
                  'j']
  reader_list = ['Olivio', 'Andres']
  
  df = pd.DataFrame(columns=['subject', 'reader', 'roi_name', 'region', 'x', 'y', 'z', 'b0', 'b10', 'b20', 'b60', 'b150', 'b300', 'b500', 'b1000'])
  
  for subject in subject_list:
    print(subject)
    # image_path = 'H:/projects/IVIM/vendor_study/data/dwi_images/Philips_1_5T_' + subject + '_DWI_FB.tif'
    image_path = 'H:/projects/IVIM/vendor_study/data/dwi_images/Philips_3T_' + subject + '_DWI_FB.tif'
    image = io.imread(image_path)
    sz, sy, sx = image.shape
    z_one_b = sz//num_b_values

    for reader in reader_list:
      print(reader)
      # roi_path = 'H:/projects/IVIM/vendor_study/data/ROIs/matlab_corrected_' + reader + '/Philips_1_5T_' + subject + '_DWI_FB.mat'
      roi_path = 'H:/projects/IVIM/vendor_study/data/ROIs/matlab_corrected_' + reader + '/Philips_3T_' + subject + '_DWI_FB.mat'
      mat = sio.loadmat(roi_path)
      num_rois = mat['roi'].shape[1]
      for roi_id in range(num_rois):
        roi_name = mat['roi'][0, roi_id][0][0]
        roi_coordinates = mat['roi'][0, roi_id][1] -1 #start at 0 for python
        region = region_map(roi_name)
        if region:
          for c in roi_coordinates:
            b_vals = image[c[2]::z_one_b, c[0], c[1]].tolist()
            record = [subject, reader, roi_name, region, c[1], c[0], c[2]]
            for b_val in b_vals:
              record.append(b_val)
            df.loc[len(df)] = record

  # df.to_pickle('./data/abdominal_data.pkl')
  # df.to_csv('./data/abdominal_data.csv', index=False)
  df.to_pickle('./data/abdominal_data_3T.pkl')
  df.to_csv('./data/abdominal_data_3T.csv', index=False)

  
  
  
