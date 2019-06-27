'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import numpy as np
import pandas as pd
from pdb import set_trace as bp

if __name__ == '__main__':

  # df = pd.read_pickle('./data/abdominal_data_LS_BP_NN.pkl')
  df = pd.read_pickle('./data/abdominal_data_LS_BP_NN_3T.pkl')
  df = df[['subject', 'reader', 'region', 'Dp_LS', 'Dt_LS', 'Fp_LS', 'Dp_BP', 'Dt_BP', 'Fp_BP', 'Dp_NN', 'Dt_NN', 'Fp_NN']]
  df_by_reader = df.groupby(['subject', 'reader', 'region'], as_index=False).mean()
  df_average = df_by_reader.groupby(['subject', 'region'], as_index=False).mean()
  df_average = df_average[['region', 'Dp_LS', 'Dt_LS', 'Fp_LS', 'Dp_BP', 'Dt_BP', 'Fp_BP', 'Dp_NN', 'Dt_NN', 'Fp_NN']]
    
  df_cv = df_average.groupby(['region']).std()/df_average.groupby(['region']).mean()
  print(np.round(df_cv.mean()*1000)/10)

  df_by_reader['new_id'] = df_by_reader['subject'] + '_' + df_by_reader['region']
  tmp = df_by_reader.pivot(index='new_id', columns='reader', values=['Dp_LS', 'Dt_LS', 'Fp_LS', 'Dp_BP', 'Dt_BP', 'Fp_BP', 'Dp_NN', 'Dt_NN', 'Fp_NN'])
  tmp.columns = ['_'.join(col).strip() for col in tmp.columns.values]
  # tmp.to_csv('./data/df_for_icc.csv', index=False)
  tmp.to_csv('./data/df_for_icc_3T.csv', index=False)
