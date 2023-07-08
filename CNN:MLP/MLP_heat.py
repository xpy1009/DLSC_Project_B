import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Data_heat import initial_condition, final_value, convert

np.random.seed(7)
torch.manual_seed(128)

def save_checkpoint(model, optimizer, scheduler, EPOCH, test_relative_l2, save_dir):
    '''save model and optimizer'''

    torch.save({
        'epoch': EPOCH,
        'best loss': test_relative_l2,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)

    
def load_checkpoint(model, optimizer, scheduler, save_dir, switch=True):
    '''load model and optimizer'''

    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    EPOCH = checkpoint['epoch']
    best_loss = checkpoint['best loss']

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print('preoptimizer loaded!')

    print('Pretrained model loaded!')
    
    return model,optimizer,scheduler, EPOCH, best_loss

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


train_size = 150
test_size = 50

res = 64
x = torch.linspace(0, 1, res)
y = torch.linspace(0, 1, res)
x_grid, y_grid = torch.meshgrid(x, y)
x_grid = x_grid.reshape(-1,1)
y_grid = y_grid.reshape(-1,1)
grid = torch.cat((x_grid, y_grid), 1)

d = 6
T = 0.1
domain_extrema = np.array([[-1, 1], [-1, 1]]) # x in [-1, 1]^2
input_x = convert(grid, domain_extrema)
n_x = input_x.shape[0]
pre_model_save_path = f'./model/MLP_heat_{d}_checkpoint.pt'
# pre_model_save_path = None
save_path = f'./model/MLP_heat_{d}_checkpoint.pt'
use_generated_data = True

print("Loading trainig data...")
if use_generated_data:
  with open(f'./Dataset/heat/input_train_{res}_d{d}.npy', 'rb') as f:
    input_train = torch.from_numpy(np.load(f))
  with open(f'./Dataset/heat/output_train_{res}_d{d}.npy', 'rb') as f_out:  
    output_train = torch.from_numpy(np.load(f_out))
  
else:    
  input_train, output_train = [], []
  for i in range(train_size):
      mu = np.random.uniform(-1, 1, d)
      input_train.append(torch.FloatTensor(np.concatenate((input_x, np.tile(mu, (n_x,1))), axis=1)))
      output_train.append(final_value(input_x, mu, T))

  input_train = torch.concatenate(input_train, axis=0).reshape(train_size, n_x, -1)
  output_train = torch.stack(output_train).reshape(train_size, n_x, -1)
  print("Saving generated training data...")
  np.save(f'./Dataset/heat/input_train_{res}_d{d}.npy', input_train)
  np.save(f'./Dataset/heat/output_train_{res}_d{d}.npy', output_train)

print("Loading validation data...")
if use_generated_data:
  with open(f'./Dataset/heat/input_test_{res}_d{d}.npy', 'rb') as f:
    input_test = torch.from_numpy(np.load(f))
  with open(f'./Dataset/heat/output_test_{res}_d{d}.npy', 'rb') as f_out:  
    output_test = torch.from_numpy(np.load(f_out))
else:    
  input_test, output_test = [], []
  for i in range(test_size):
      mu = np.random.uniform(-1, 1, d)
      input_test.append(torch.FloatTensor(np.concatenate((input_x, np.tile(mu, (n_x,1))), axis=1)))
      output_test.append(final_value(input_x, mu, T))

  input_test = torch.concatenate(input_test, axis=0).reshape(test_size, n_x, -1)
  output_test = torch.stack(output_test).reshape(test_size, n_x, -1)
  print("Saving generated testing data...")
  np.save(f'./Dataset/heat/input_test_{res}_d{d}.npy', input_test)
  np.save(f'./Dataset/heat/output_test_{res}_d{d}.npy', output_test)

batch_size = 32
training_set = DataLoader(torch.utils.data.TensorDataset(input_train, output_train), batch_size=batch_size, shuffle=True)
testing_set = DataLoader(torch.utils.data.TensorDataset(input_test, output_test), batch_size=50, shuffle=True)
model = MLPnet(2+d, 150, 1)
n_epoch = 200
learning_rate = 0.01
start_epoch = 0

step_size = 3
gamma = 0.8
best_loss= np.inf

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

if pre_model_save_path is not None:
    model, optimizer, scheduler, start_epoch, best_loss=load_checkpoint(model,optimizer,scheduler,pre_model_save_path,switch=None)
    print("Sart epoch:", start_epoch)
    print("Best loss:", best_loss)

for param_group in optimizer.param_groups:
    print(param_group['lr'])

loss_criterion = nn.MSELoss()
freq_print = 1
history = []
print("Start training...")

for epoch_c in range(n_epoch):
  if pre_model_save_path:
    epoch = epoch_c + start_epoch
  else:
    epoch = epoch_c 
  train_mse = 0.0
  for j, data in enumerate(training_set):
      optimizer.zero_grad()
      inp_train, output_train = data
      predicted = model(inp_train)
      loss = loss_criterion(predicted, output_train)
      loss.backward()
      train_mse += loss.item()
      optimizer.step()
  train_mse /= len(training_set)
  history.append(train_mse)
  scheduler.step()    

  with torch.no_grad():
        model.eval()
        test_relative_l2 = 0.0
        for step, data in enumerate(testing_set):
            input_batch, output_batch = data
            output_pred_batch = model(input_batch)
            loss_f = (torch.mean((output_pred_batch - output_batch) ** 2) / torch.mean(output_batch ** 2)) ** 0.5 * 100
            test_relative_l2 += loss_f.item()
        test_relative_l2 /= len(testing_set)
        # save model
        if  test_relative_l2 < best_loss:
            save_checkpoint(model, optimizer, scheduler, epoch, test_relative_l2, save_path)
            print('successfully save the model')
            best_loss = test_relative_l2

  if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss:", train_mse, " ######### Relative L2 Test Norm:", test_relative_l2)          


print('Final Loss: ', history[-1])

plt.figure(dpi=150)
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(start_epoch, len(history) + start_epoch), history, label="Train Loss")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig(f'loss_MLP_{res}_{d}.png', dpi=150)

test_pred = model(input_test)
loss_eval = loss_criterion(test_pred, output_test)
print("Validation loss:", loss_eval.item())

inputs_x = input_x 
mu = np.random.uniform(-1, 1, d)
ground_truth = u_T = final_value(inputs_x, mu, T)
inputs = np.concatenate((inputs_x, np.tile(mu, (n_x,1))), axis=1)
inputs = torch.FloatTensor(inputs)
predicted = model(inputs)
loss_criterion(predicted, ground_truth.reshape(n_x, 1))
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
plt.savefig(f'res_MLP_{res}_{d}.png', dpi=150)