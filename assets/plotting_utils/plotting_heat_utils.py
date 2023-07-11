import numpy as np
import torch
import matplotlib.pyplot as plt

d = 6
res = 64
# replace with desired plotting data file path
data = np.load(f"./heat_plotting_data_d{d}.npy")
print(data[0])
input_x = data[:, :2]
mu = data[0, 2:2+d]
u_0 = data[:, 2+d]
u_T = data[:, 3+d]

print("Inputs:", input_x.shape, "\n mu:", mu.shape, "\n u_0:", u_0.shape, "\n u_T:", u_T.shape)

fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
im1 = axs[0].contourf(input_x[:, 0].reshape(res,res), input_x[:, 1].reshape(res,res), u_0.reshape(res,res), levels = 64, cmap="jet")
axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")
axs[0].set_title("u_0")
plt.colorbar(im1, ax=axs[0])
axs[0].grid(True, which="both", ls=":")
im2 = axs[1].contourf(input_x[:, 0].reshape(res,res), input_x[:, 1].reshape(res,res), u_T.reshape(res,res), levels = 64, cmap="jet")
axs[1].set_xlabel("x1")
axs[1].set_ylabel("x2")
axs[1].set_title("u_T")
plt.colorbar(im2, ax=axs[1])
axs[1].grid(True, which="both", ls=":")
plt.show()