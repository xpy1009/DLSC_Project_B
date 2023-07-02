import torch
import matplotlib.pyplot as plt
from Data_wave import initial_condition, final_value, convert
import numpy as np

# tensor-product (stucture mesh)
res = 32
x = torch.linspace(0, 1, res)
y = torch.linspace(0, 1, res)
x_grid, y_grid = torch.meshgrid(x, y)
x_grid = x_grid.reshape(-1,1)
y_grid = y_grid.reshape(-1,1)
grid = torch.cat((x_grid, y_grid), 1)

# Calculating
d = 24
T = 5
domain_extrema = np.array([[0, 1], [0, 1]]) # x in [-1, 1]^2
input_x = convert(grid, domain_extrema)
mu = np.random.uniform(-1, 1, (d,d))
u_0 = initial_condition(input_x, mu)
u_T = final_value(input_x, mu, T)

# plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
im1 = axs[0].contourf(input_x[:,0].reshape(res,res),input_x[:,1].reshape(res,res),u_0.reshape(res,res),levels = 64,cmap="jet")
axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")
plt.colorbar(im1, ax=axs[0])
axs[0].grid(True, which="both", ls=":")
axs[0].set_title("u_0")
axs[0].set(aspect='equal')
im2 = axs[1].contourf(input_x[:,0].reshape(res,res),input_x[:,1].reshape(res,res),u_T.reshape(res,res),levels = 64,cmap="jet")
axs[1].set_xlabel("x1")
axs[1].set_ylabel("x2")
plt.colorbar(im2, ax=axs[1])
axs[1].grid(True, which="both", ls=":")
axs[1].set_title("u_T")
axs[1].set(aspect='equal')
plt.savefig('res.png', dpi=150)

# Heat Diffusion
# train_size = 100
# test_size = 100
# d_size = train_size + test_size
# u0s = np.empty([d_size, 96, 96, 3])
# uTs = np.empty([d_size, 96, 96])
# for d in range(1,7):
#     for i in range(d_size):
#         mu = np.random.uniform(-1, 1, d)
#         u_0 = initial_condition(input_x, mu)
#         u_T = final_value(input_x, mu, T)
#         u0s[i,:,:,0] = u_0.reshape(96,96)
#         u0s[i,:,:,1] = input_x[:,0].reshape(96,96)
#         u0s[i,:,:,2] = input_x[:,1].reshape(96,96)
#         uTs[i, :] = u_T.reshape(96,96)
#     np.save("u0s_d" + str(d),u0s)
#     np.save("uTs_d" + str(d),uTs)

# Wave equation
# train_size = 100
# test_size = 100
# d_size = train_size + test_size
# u0s = np.empty([d_size, 64, 64, 3])
# uTs = np.empty([d_size, 64, 64])
# for i in range(d_size):
#     mu = np.random.uniform(-1, 1, [d,d])
#     u_0 = initial_condition(input_x, mu)
#     u_T = final_value(input_x, mu, T)
#     u0s[i,:,:,0] = u_0.reshape(64,64)
#     u0s[i,:,:,1] = input_x[:,0].reshape(64,64)
#     u0s[i,:,:,2] = input_x[:,1].reshape(64,64)
#     uTs[i, :] = u_T.reshape(64,64)
#
# np.save("u0s_d" + str(d),u0s)
# np.save("uTs_d" + str(d),uTs)
