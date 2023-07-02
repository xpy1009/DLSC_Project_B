import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Data import initial_condition, final_value, convert

np.random.seed(7)
torch.manual_seed(128)

class MLPnet(nn.Module):

  def __init__(self, input_dim, hidden_size, output_dim):
    super(MLPnet, self).__init__()

    self.model = nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_dim)
    )

  def forward(self, x):
    return self.model(x)

n_x = 128
d = 2
T = 0.1
train_size = 100
test_size = 100
# n_mu = 20
domain_extrema = torch.tensor([[-1, 1], [-1, 1]]) # x in [-1, 1]^2
soboleng = torch.quasirandom.SobolEngine(dimension=domain_extrema.shape[0])
input_x = convert(soboleng.draw(n_x), domain_extrema)

print("Loading trainig data...")
input_train, output_train = [], []
for i in range(train_size):
    mu = np.random.uniform(-1, 1, d)
    # print(mu)
    input_train.append(torch.FloatTensor(np.concatenate((input_x, np.tile(mu, (n_x,1))), axis=1)))
    output_train.append(final_value(input_x, mu, T))

input_train = torch.concatenate(input_train, axis=0).reshape(train_size, n_x, -1)
output_train = torch.stack(output_train).reshape(train_size, n_x, -1)

print("Loading validation data...")
input_test, output_test = [], []
for i in range(test_size):
    mu = np.random.uniform(-1, 1, d)
    # print(mu)
    input_test.append(torch.FloatTensor(np.concatenate((input_x, np.tile(mu, (n_x,1))), axis=1)))
    output_test.append(final_value(input_x, mu, T))

input_test = torch.concatenate(input_test, axis=0).reshape(test_size, n_x, -1)
output_test = torch.stack(output_test).reshape(test_size, n_x, -1)

training_set = DataLoader(torch.utils.data.TensorDataset(input_train, output_train), batch_size=n_x, shuffle=True)
# testing_set = DataLoader(torch.utils.data.TensorDataset(input_test, output_test), batch_size=n_x, shuffle=True)
model = MLPnet(2+d, 150, 1)
n_epoch = 1
optimizer = optim.LBFGS(model.parameters(),
                              lr=float(0.5),
                              max_iter=10000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
loss_criterion = nn.MSELoss()
history = []
print("Start training...")
for epoch in range(n_epoch):
  for j, data in enumerate(training_set):
      if j != 0 and j % 100 == 0:
         print(j, "loss: ", history[-1])
      inp_train, output_train = data
      def closure():
        predicted = model(inp_train)
        loss = loss_criterion(predicted, output_train)
        optimizer.zero_grad()
        loss.backward()

        history.append(loss.item())
        return loss
      optimizer.step(closure=closure)

print('Final Loss: ', history[-1])

plt.figure(dpi=150)
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1, len(history) + 1), history, label="Train Loss")
plt.xscale("log")
plt.legend()
plt.show()

test_pred = model(input_test)
loss_eval = loss_criterion(test_pred, output_test)
print("Validation loss:", loss_eval.item())

inputs_x = soboleng.draw(10000)
inputs_x = convert(inputs_x, domain_extrema)
mu = np.random.uniform(-1, 1, d)
ground_truth = u_T = final_value(inputs_x, mu, T)
inputs = np.concatenate((inputs_x, np.tile(mu, (10000,1))), axis=1)
inputs = torch.FloatTensor(inputs)
predicted = model(inputs)
loss_criterion(predicted, ground_truth.reshape(10000, 1))
fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
im1 = axs[0].scatter(inputs_x[:, 1], inputs_x[:, 0], c=predicted.detach().numpy(), cmap="jet")
axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")
axs[0].set_title("Predicted")
plt.colorbar(im1, ax=axs[0])
axs[0].grid(True, which="both", ls=":")
im2 = axs[1].scatter(inputs_x[:, 1], inputs_x[:, 0], c=ground_truth, cmap="jet")
axs[1].set_xlabel("x1")
axs[1].set_ylabel("x2")
axs[1].set_title("Ground Truth")
plt.colorbar(im2, ax=axs[1])
axs[1].grid(True, which="both", ls=":")
plt.savefig('res_MLP.png', dpi=150)