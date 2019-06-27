'''
Jan 2019 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pdb import set_trace as bp

if __name__ == '__main__':

  # df = pd.read_pickle('./data/abdominal_data_LS_BP_NN.pkl')
  df = pd.read_pickle('./data/abdominal_data_LS_BP_NN_3T.pkl')
  
  # df = df[['subject', 'reader', 'region', 'Dp_LS', 'Dt_LS', 'Fp_LS', 'Dp_BP', 'Dt_BP', 'Fp_BP', 'Dp_NN', 'Dt_NN', 'Fp_NN']]
  df_by_reader = df.groupby(['subject', 'reader', 'region'], as_index=False).mean()
  df_average = df_by_reader.groupby(['subject', 'region'], as_index=False).mean()
  df_average = df_average[['region', 'Dp_LS', 'Dt_LS', 'Fp_LS', 'Dp_BP', 'Dt_BP', 'Fp_BP', 'Dp_NN', 'Dt_NN', 'Fp_NN']]
  
  # wide to long
  df_average['id'] = df_average.index
  df_average = pd.wide_to_long(df_average, ['Dp', 'Dt', 'Fp'], i='id', j='algorithm', sep='_', suffix='\w+')
  df_average.reset_index(level=['algorithm'], inplace=True)

  # plot
  fg, ax = plt.subplots(3, 1, sharex='all')
  
  sns.boxplot(data=df_average, hue='algorithm', x='region', y='Dt', linewidth=0.5, width=0.5, fliersize=1,
              order = ['left_liver_lobe', 'right_liver_lobe', 'pancreas', 'spleen', 'renal_cortex', 'renal_medulla'], ax=ax[0])
  ax[0].set(xlabel='', ylabel=r'$D_t$ [mm$^2$/sec]')
  ax[0].legend(loc='upper left', fontsize='small', ncol=3)
  ax[0].get_legend().get_texts()[2].set_text('DNN')

  df_average['Fp'] = df_average['Fp']*100  
  sns.boxplot(data=df_average, hue='algorithm', x='region', y='Fp', linewidth=0.5, width=0.5, fliersize=1,
              order = ['left_liver_lobe', 'right_liver_lobe', 'pancreas', 'spleen', 'renal_cortex', 'renal_medulla'], ax=ax[1])
  ax[1].set(xlabel='', ylabel=r'$F_p$ [%]')
  ax[1].get_legend().remove()
  
  sns.boxplot(data=df_average, hue='algorithm', x='region', y='Dp', linewidth=0.5, width=0.5, fliersize=1,
              order = ['left_liver_lobe', 'right_liver_lobe', 'pancreas', 'spleen', 'renal_cortex', 'renal_medulla'], ax=ax[2])
  ax[2].set(xlabel='', ylabel=r'$D_p$ [mm$^2$/sec]')
  ax[2].set_xticklabels(['Left\nLiver Lobe', 'Right\nLiver Lobe', 'Pancreas', 'Spleen', 'Renal\nCortex', 'Renal\nMedulla'])
  ax[2].set_ylim([0, 0.3])
  ax[2].get_legend().remove()  
  
  fg.tight_layout()
  # fg.savefig('./fig/abdominal_boxplots.pdf', bbox_inches='tight')
  # fg.savefig('./fig/abdominal_boxplots.tif', dpi=fg.dpi, bbox_inches='tight')
  fg.savefig('./fig/abdominal_boxplots_3T.pdf', bbox_inches='tight')
  fg.savefig('./fig/abdominal_boxplots_3T.tif', dpi=fg.dpi, bbox_inches='tight')
  
