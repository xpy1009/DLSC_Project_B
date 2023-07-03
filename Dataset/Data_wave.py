import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

np.random.seed(7)
torch.manual_seed(128)

# x: [n, 2], grid point coordinates
# mu: [d] Uniform([-1, -1]^d)
# return: [n] initial value for each grid point
def initial(x,a):
  K = a.shape[0]
  u_0 = torch.zeros(1)

  for i in range(1, K+1, 1):
    for j in range(1, K+1, 1):
      u_0 += a[i-1,j-1] * (i*i + j*j) ** (-1) * np.sin(np.pi * i * x[0]) * np.sin(np.pi * j * x[1])

  return u_0 * np.pi / (K * K)

def initial_condition(x, a):
  n = x.shape[0]
  f_xy = torch.zeros(n)
  for i in range(n):
    f_xy[i] = initial(x[i,:], a)

  return f_xy


def initial_condition_fast(x, mu):
  d = mu.shape[0]
  n = x.shape[0]
  u_0 = torch.zeros((n))

  x_init = np.repeat(x, d*d, axis=0) * np.pi
  x_init = x_init.reshape(n, d*d, -1) 
  count = 0
  for i in range(1, d+1, 1):
    for j in range(1, d+1, 1):
      x_init[:, count, 0] *= i
      x_init[:, count, 1] *= j
      count += 1    
  u = np.sin(x_init)
  count = 0
  for i in range(1, d+1, 1):
    for j in range(1, d+1, 1):
      u_0[:] += u[:, count, 0] * u[:, count, 1]* mu[i-1, j-1] * (i*i + j*j) ** (-1)
      count += 1

  return u_0 * np.pi / (d * d)


def final(x,a,T):
  K = a.shape[0]
  u_xyt = torch.zeros(1)

  for i in range(1, K + 1, 1):
    for j in range(1, K + 1, 1):
      u_xyt += a[i - 1, j - 1] * (i * i + j * j) ** (-1) * np.sin(np.pi * i * x[0]) * np.sin(np.pi * j * x[1]) * np.cos(0.1 * np.pi * T * np.sqrt(i*i + j*j))

  return u_xyt * np.pi / (K * K)

def final_value(x, a, T):
  n = x.shape[0]
  u_xyt = torch.zeros(n)
  for i in range(n):
    u_xyt[i] = final(x[i, :], a, T)

  return u_xyt

def final_value_fast(x, mu, T):
  d = mu.shape[0]
  n = x.shape[0]
  u_T = torch.zeros((n))

  x_final = np.repeat(x, d*d, axis=0) * np.pi
  x_final = x_final.reshape(n, d*d, -1) 
  count = 0
  for i in range(1, d+1, 1):
    for j in range(1, d+1, 1):
      x_final[:, count, 0] *= i
      x_final[:, count, 1] *= j
      count += 1    
  u = np.sin(x_final)
  count = 0
  for i in range(1, d+1, 1):
    for j in range(1, d+1, 1):
      u_T[:] += u[:, count, 0] * u[:, count, 1]* mu[i-1, j-1] * (i*i + j*j) ** (-1) * np.cos(0.1 * np.pi * T * np.sqrt(i*i + j*j))
      count += 1

  return u_T * np.pi / (d * d)    

def convert(tens, domain_extrema):
  assert (tens.shape[1] == domain_extrema.shape[0])
  return tens * (domain_extrema[:, 1] - domain_extrema[:, 0]) + domain_extrema[:, 0]