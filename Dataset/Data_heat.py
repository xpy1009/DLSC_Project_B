import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

np.random.seed(7)
torch.manual_seed(128)

# x: [n, 2], grid point coordinates
# mu: [d] Uniform([-1, -1]^d)
# return: [n] initial value for each grid point
def initial_condition(x, mu):
  d = mu.shape[0]
  n = x.shape[0]
  u_0 = torch.zeros((n))

  x_init = np.repeat(x, d, axis=0) * np.pi
  x_init = x_init.reshape(n, d, -1)
  for i in range(1, d+1, 1):
    x_init[:, i-1] *= i
  u = np.sin(x_init)
  for i in range(1, d+1, 1):
    u_0[:] += -u[:, i-1, 0] * u[:, i-1, 1]* mu[i-1] / i**0.5

  return u_0/d

# x: [n, 2], grid point coordinates
# mu: [d] Uniform([-1, -1]^d)
# return : [n] final value for each grid point
def final_value(x, mu, T):
  d = mu.shape[0]
  n = x.shape[0]
  u_T = torch.zeros((n))

  x_final = np.repeat(x, d, axis=0) * np.pi
  x_final = x_final.reshape(n, d, -1)
  for i in range(1, d+1, 1):
    x_final[:, i-1] *= i
  u = np.sin(x_final)
  for i in range(1, d+1, 1):
    u_T[:] += -u[:, i-1, 0] * u[:, i-1, 1]* mu[i-1] / i**0.5 * np.exp(-2.*(i*np.pi)**2*T)

  return u_T

def convert(tens, domain_extrema):
  assert (tens.shape[1] == domain_extrema.shape[0])
  return tens * (domain_extrema[:, 1] - domain_extrema[:, 0]) + domain_extrema[:, 0]