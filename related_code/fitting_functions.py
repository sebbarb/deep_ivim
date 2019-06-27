'''
Mar 2018 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/deep-learning/ivim
'''

from scipy.optimize import curve_fit, minimize
import numpy as np
from hyperparameters import Hyperparameters as hp
import matplotlib.pyplot as plt
from scipy import stats
from pdb import set_trace as bp


def ivim(b, Dp, Dt, Fp):
  return Fp*np.exp(-b*Dp) + (1-Fp)*np.exp(-b*Dt)


def order(Dp, Dt, Fp):
  if Dp < Dt:
    Dp, Dt = Dt, Dp
    Fp = 1-Fp
  return Dp, Dt, Fp  


def fit_segmented(b, x_dw):
  try:
    high_b = b[b>=250]
    high_x_dw = x_dw[b>=250]
    bounds = (0, 1)
    # bounds = ([0, 0.4], [0.005, 1])
    params, _ = curve_fit(lambda high_b, Dt, int : int*np.exp(-high_b*Dt), high_b, high_x_dw, p0=(0.001, 0.9), bounds=bounds)
    Dt, Fp = params[0], 1-params[1]
    x_dw_remaining = x_dw - (1-Fp)*np.exp(-b*Dt)
    bounds = (0, 1)
    # bounds = (0.01, 0.3)
    params, _ = curve_fit(lambda b, Dp : Fp*np.exp(-b*Dp), b, x_dw_remaining, p0=(0.01), bounds=bounds)
    Dp = params[0]
    return order(Dp, Dt, Fp)
  except:
    return 0., 0., 0.


def fit_least_squares(b, x_dw):
  try:
    bounds = (0, 1)
    # bounds = ([0.01, 0, 0], [0.3, 0.005, 0.6])
    params, _ = curve_fit(ivim, b, x_dw, p0=[0.01, 0.001, 0.1], bounds=bounds)
    Dp, Dt, Fp = params[0], params[1], params[2]
    return order(Dp, Dt, Fp)
  except:
    return fit_segmented(b, x_dw)


def empirical_neg_log_prior(Dp0, Dt0, Fp0):
  #Dp0, Dt0, Fp0 are flattened arrays
  Dp_valid = (1e-8 < np.nan_to_num(Dp0)) & (np.nan_to_num(Dp0) < 1-1e-8)
  Dt_valid = (1e-8 < np.nan_to_num(Dt0)) & (np.nan_to_num(Dt0) < 1-1e-8)
  Fp_valid = (1e-8 < np.nan_to_num(Fp0)) & (np.nan_to_num(Fp0) < 1-1e-8)
  valid = Dp_valid & Dt_valid & Fp_valid
  Dp0, Dt0, Fp0 = Dp0[valid], Dt0[valid], Fp0[valid]
  Dp_shape, _, Dp_scale = stats.lognorm.fit(Dp0, floc=0)
  Dt_shape, _, Dt_scale = stats.lognorm.fit(Dt0, floc=0)
  Fp_a, Fp_b, _, _ = stats.beta.fit(Fp0, floc=0, fscale=1)
  def neg_log_prior(p):
    Dp, Dt, Fp, = p[0], p[1], p[2]
    if (Dp < Dt):
      return 1e8
    else:
      eps = 1e-8
      Dp_prior = stats.lognorm.pdf(Dp, Dp_shape, scale=Dp_scale)
      Dt_prior = stats.lognorm.pdf(Dt, Dt_shape, scale=Dt_scale)
      Fp_prior = stats.beta.pdf(Fp, Fp_a, Fp_b)
      return -np.log(Dp_prior+eps) - np.log(Dt_prior+eps) - np.log(Fp_prior+eps)
  return neg_log_prior


def neg_log_likelihood(p, b, x_dw):
  return 0.5*(len(b)+1)*np.log(np.sum((ivim(b, p[0], p[1], p[2])-x_dw)**2)) #0.5*sum simplified


def neg_log_posterior(p, b, x_dw, neg_log_prior):
  return neg_log_likelihood(p, b, x_dw) + neg_log_prior(p)


def fit_bayesian(b, x_dw, neg_log_prior):
  try:
    bounds = [(0, 1), (0, 1), (0, 1)]
    # bounds = [(0.01, 0.3), (0, 0.005), (0, 0.6)]
    params = minimize(neg_log_posterior, x0=[0.01, 0.001, 0.1], args=(b, x_dw, neg_log_prior), bounds=bounds)
    if not params.success:
      # print(params.message)
      raise(params.message)
    Dp, Dt, Fp = params.x[0], params.x[1], params.x[2]
    return order(Dp, Dt, Fp)
  except:
    return fit_least_squares(b, x_dw)
      

if __name__ == '__main__':
  # noise 15
  # Dp_truth = array([0.04775578])
  # Dt_truth = array([0.00099359])
  # Fp_truth = array([0.17606327])
  x_dw = np.array([0.92007383, 0.89169505, 0.79940185, 0.73149624, 0.62961453, 0.50390379, 0.30776271])

  # noise 150
  # Dp_truth = array([0.04775578])
  # Dt_truth = array([0.00099359])
  # Fp_truth = array([0.17606327])
  # x_dw = np.array([0.85389307, 1.03929503, 0.91976122, 0.96119655, 0.80588596, 0.53290429, 0.34947694])

  b_values_no0 = np.array(hp.b_values[1:])
  t = fit_segmented(b_values_no0, x_dw)
  print(t)
  t = fit_least_squares(b_values_no0, x_dw)
  print(t)
  t = fit_bayesian(b_values_no0, x_dw, np.array([0.01, 0.001, 0.1]), np.array([[10000, 0, 0], [0, 10000, 0], [0, 0, 10000]]))
  print(t)  
  