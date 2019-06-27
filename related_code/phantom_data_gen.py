'''
Mar 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

from hyperparameters import Hyperparameters as hp
import numpy as np
from pdb import set_trace as bp

if __name__ == '__main__':
  #generate random parameter vectors
  rg = np.random.RandomState(hp.rs_1)
    
  Dp = rg.uniform(hp.Dp_min, hp.Dp_max, (hp.num_samples,1))
  Dt = rg.uniform(hp.Dt_min, hp.Dt_max, (hp.num_samples,1))
  Fp = rg.uniform(hp.Fp_min, hp.Fp_max, (hp.num_samples,1))
  noise_sd = rg.uniform(hp.noise_sd_min, hp.noise_sd_max, (hp.num_samples,1))

  #generate dw signals and store them in X_dw
  b_values = np.array(hp.b_values)
  num_b_values = len(b_values)
  X_dw = np.zeros((hp.num_samples, num_b_values))
  rg = np.random.RandomState(hp.rs_2)
  for i in range(0,hp.num_samples):
      dw_signal = Fp[i]*np.exp(-b_values*Dp[i]) + (1-Fp[i])*np.exp(-b_values*Dt[i])
      if hp.add_noise:
          #add noise
          dw_signal_scaled = hp.S0*dw_signal
          noise_real = rg.normal(0, noise_sd[i], (1,num_b_values))
          noise_imag = rg.normal(0, noise_sd[i], (1,num_b_values))
          dw_signal_scaled_noisy = np.sqrt(np.power(dw_signal_scaled+noise_real,2)+np.power(noise_imag,2))
          S0_noisy = np.mean(dw_signal_scaled_noisy[0,b_values==0])
          dw_signal_noisy = dw_signal_scaled_noisy/S0_noisy
          X_dw[i,] = dw_signal_noisy
      else:
          X_dw[i,] = dw_signal

  X_dw = X_dw[:,1:]

  #save
  if hp.add_noise:
      np.savez(hp.data_file, Dp=Dp, Dt=Dt, Fp=Fp, X_dw=X_dw, add_noise=hp.add_noise, noise_sd=noise_sd)
  else:
      np.savez(hp.data_file, Dp=Dp, Dt=Dt, Fp=Fp, X_dw=X_dw, add_noise=hp.add_noise)

