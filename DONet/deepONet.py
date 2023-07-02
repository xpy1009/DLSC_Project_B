import numpy as np
import torch
import matplotlib.pyplot as plt
import deepxde as dde
from Data import initial_condition, final_value, convert

np.random.seed(7)
torch.manual_seed(128)

n_x = 8192
d = 6
T = 0.1
domain_extrema = torch.tensor([[-1, 1], [-1, 1]]) # x in [-1, 1]^2
soboleng = torch.quasirandom.SobolEngine(dimension=domain_extrema.shape[0])
input_x = convert(soboleng.draw(n_x), domain_extrema).numpy()

train_size = 100
test_size = 100
d_size = train_size + test_size
u0s = np.empty([d_size, n_x])
uTs = np.empty([d_size, n_x])

for i in range(d_size):
  mu = np.random.uniform(-1, 1, d)
  u_0 = initial_condition(input_x, mu)
  u_T = final_value(input_x, mu, T)
  u0s[i, :] = u_0
  uTs[i, :] = u_T

X_train = (u0s[:train_size, :].astype(np.float32), input_x.astype(np.float32))
y_train = uTs[:train_size, :].astype(np.float32)

X_test = (u0s[train_size:, :].astype(np.float32), input_x.astype(np.float32))
y_test = uTs[train_size:, :].astype(np.float32)

data = dde.data.TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

m = n_x
dim_x = 2
net = dde.nn.DeepONetCartesianProd(
    [m, 40, 40],
    [dim_x, 40, 40],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and train.
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
losshistory, train_state = model.train(iterations=10000)

# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.savefig('loss.png', dpi=150)

mu = np.random.uniform(-1, 1, d)
u_0 = initial_condition(input_x, mu).numpy()
u_T = final_value(input_x, mu, T).numpy()
X_eval = (u_0.reshape(1,-1).astype(np.float32), input_x.astype(np.float32))
pred = model.predict(X_eval).flatten()
print("MSE loss:", np.mean(np.square((pred-u_T))))

fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
im1 = axs[0].scatter(input_x[:, 1], input_x[:, 0], c=pred, cmap="jet")
axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")
plt.colorbar(im1, ax=axs[0])
axs[0].grid(True, which="both", ls=":")
axs[0].set_title("pred")
im2 = axs[1].scatter(input_x[:, 1], input_x[:, 0], c=u_T, cmap="jet")
axs[1].set_xlabel("x1")
axs[1].set_ylabel("x2")
plt.colorbar(im2, ax=axs[1])
axs[1].grid(True, which="both", ls=":")
axs[1].set_title("ground truth")
plt.savefig('res.png', dpi=150)
