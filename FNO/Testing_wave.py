import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter



def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['elu']:
        return nn.ELU()
    elif name in ['mish']:
        return nn.Mish()
    else:
        raise ValueError('Unknown activation function')


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)

    
def load_checkpoint(model, optimizer, scheduler, save_dir, switch=1):
    '''load model and optimizer'''

    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if switch:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('preobtimizer loaded!')

    print('Pretrained model loaded!')
    
    return model,optimizer,scheduler

################################################################
#  2d fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, fno_architecture, device=None, padding_frac= 1/4):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = fno_architecture["modes"]
        self.modes2 = fno_architecture["modes"]
        self.width = fno_architecture["width"]
        self.n_layers = fno_architecture["n_layers"]
        self.retrain_fno = fno_architecture["retrain_fno"]

        torch.manual_seed(self.retrain_fno)
        # self.padding = 9 # pad the domain if input is non-periodic
        self.padding_frac = padding_frac
        self.fc0 = nn.Linear(1, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.to(device)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1_padding = int(round(x.shape[-1] * self.padding_frac))
        x2_padding = int(round(x.shape[-2] * self.padding_frac))
        x = F.pad(x, [0, x1_padding, 0, x2_padding])

        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):

            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = F.gelu(x)
        x = x[..., :-x1_padding, :-x2_padding]

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    

torch.manual_seed(0)
np.random.seed(0)

# Define tensorboard class
# writer = SummaryWriter(log_dir="Diffusion_d5")
# tags = ["train_loss", "data_error","learning_rate"]

pre_model_save_path = '../../model/Wave/FNO/FN0_K24_checkpoint1000.pt'



n_train =512
n_test = 128

res = 64

u_0 = torch.from_numpy(np.load('../../Dataset/wave/64_64/u0s_K24.npy').reshape(n_train+n_test,res,res,3)).float()
u_T = torch.from_numpy(np.load('../../Dataset/wave/64_64/uTs_K24.npy').reshape(n_train+n_test,res,res,1)).float()
print("shape of u_0:", u_0.shape)
print("shape of u_T:", u_T.shape)

#------------rescal inverse-------------------#
u_0_scale = torch.from_numpy(np.load('../../Dataset/wave/64_64/u0s_K24.npy').reshape(n_train+n_test,64,64,3)).float()
u_T_scale = torch.from_numpy(np.load('../../Dataset/wave/64_64/uTs_K24.npy').reshape(n_train+n_test,64,64,1)).float()
# scale data with normalization
min_vals_input = u_0_scale[0:n_train,:,:,0].min()
max_vals_input = u_0_scale[0:n_train,:,:,0].max()
min_vals_output = u_T_scale[0:n_train,:,:,0].min()
max_vals_output = u_T_scale[0:n_train,:,:,0].max()


tr_inputs = (u_0[0:n_train,:,:,0:1] - min_vals_input) / (max_vals_input - min_vals_input)
tr_label = (u_T[0:n_train,:,:,0:1] - min_vals_output) / (max_vals_output - min_vals_output)

ts_inputs = (u_0[n_train:,:,:,0:1] - min_vals_input) / (max_vals_input - min_vals_input)
ts_label = (u_T[n_train:,:,:,0:1] - min_vals_output) / (max_vals_output - min_vals_output)


batch_size = 50
training_set = DataLoader(TensorDataset(tr_inputs,tr_label), batch_size=batch_size, shuffle=True)
testing_set = DataLoader(TensorDataset(ts_inputs, ts_label), batch_size=50, shuffle=False)

learning_rate = 0.01

epochs = 100
step_size = 10
gamma = 0.5

# model
fno_architectures = {
  "modes": 16,
  "width": 32,
  "n_layers": 4,
  "retrain_fno": 100
}
fno = FNO2d(fno_architectures)
optimizer = Adam(fno.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

if pre_model_save_path:
    fno, optimizer, scheduler=load_checkpoint(fno,optimizer,scheduler,pre_model_save_path,switch=None)

for param_group in optimizer.param_groups:
    print(param_group['lr'])

l = torch.nn.MSELoss()

test_relative_l2 = 0.0

for step, (input_batch, output_batch) in enumerate(testing_set):
    output_pred_batch = fno(input_batch).squeeze(2)
    output_pred_batch = output_pred_batch * (max_vals_output - min_vals_output)  + min_vals_output
    output_batch = output_batch * (max_vals_output - min_vals_output)  + min_vals_output
    loss_f = (torch.mean((output_pred_batch - output_batch) ** 2) / torch.mean(output_batch ** 2)) ** 0.5 * 100
    test_relative_l2 += loss_f
test_relative_l2 /= len(testing_set)
print("test_mse = ", test_relative_l2)

print(output_batch.shape)
print(output_pred_batch.shape)
fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
#im1 = axs[0].contourf(u_0[0,:,:,1],u_0[0,:,:,2],output_batch[5,:,:,0],levels = 64,cmap="jet")
im1 = axs[0].imshow(input_batch[23,:,:,0],cmap="jet")
axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")
plt.colorbar(im1, ax=axs[0])
axs[0].grid(True, which="both", ls=":")
axs[0].set_title("true")
axs[0].set(aspect='equal')
#im2 = axs[1].contourf(u_0[0,:,:,1],u_0[0,:,:,2],output_pred_batch.detach()[5,:,:,0],levels = 64,cmap="jet")
im2 = axs[1].imshow(output_pred_batch.detach()[23,:,:,0],cmap="jet")
axs[1].set_xlabel("x1")
axs[1].set_ylabel("x2")
plt.colorbar(im2, ax=axs[1])
axs[1].grid(True, which="both", ls=":")
axs[1].set_title("prediction")
axs[1].set(aspect='equal')
plt.savefig('test.png', dpi=150)