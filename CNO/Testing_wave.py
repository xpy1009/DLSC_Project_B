import copy
import json
import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from Problems.Benchmarks import Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, \
    ShearLayer, Heatdiffusion

if len(sys.argv) == 1:

    training_properties = {
        "learning_rate": 0.0005,
        "weight_decay": 1e-10,
        "scheduler_step": 30,
        "scheduler_gamma": 0.95,
        "epochs": 1000,
        "batch_size": 16,
        "exp": 1,  # Do we use L1 or L2 errors? Default: L1
        "training_samples": 150,  # How many training samples?
    }
    model_architecture_ = {

        # ----------------------------------------------------------------------
        # Parameters to be chosen with model selection:

        "N_layers": 6,  # Number of (D) + (U) layers. In our experiments, N_layers must be even.
        "kernel_size": 3,  # Kernel size.
        "channel_multiplier": 32,  # Parameter d_e (how the number of channels changes)

        "N_res": 6,  # Number of (R) blocks.
        "res_len": 2,  # Coefficienr r in (R) definition.

        # ----------------------------------------------------------------------
        # Parameters that depend on the problem:

        "in_size": 64,  # Resolution of the computational grid
        "retrain": 4,  # Random seed

        # ----------------------------------------------------------------------
        # We fix the following parameters:

        # Filter properties:
        "cutoff_den": 2.0001,  #
        "lrelu_upsampling": 2,  # Coefficient N_{\sigma}. Default is 2.
        "half_width_mult": 1,  # Coefficient c_h. Default is 1
        "filter_size": 6,  # 2xfilter_size is the number of taps N_{tap}. Default is 6.
        "radial_filter": 0,  # Is the filter radially symmetric? Default is 0 - NO.

        "FourierF": 0,  # Number of Fourier Features in the input channels. Default is 0.

        # ----------------------------------------------------------------------
    }

    #   "which_example" can be

    #   poisson             : Poisson equation
    #   wave_0_5            : Wave equation
    #   cont_tran           : Smooth Transport
    #   disc_tran           : Discontinuous Transport
    #   allen               : Allen-Cahn equation
    #   shear_layer         : Navier-Stokes equations
    #   airfoil             : Compressible Euler equations
    #   Heatdiffusion       : Heat diffusion

    which_example = "wave_0_5"

    # Save the models here:
    folder = "../../model/Wave/"+"CNO_"+which_example

else:

    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1]
    training_properties = json.loads(sys.argv[2].replace("\'", "\""))
    model_architecture_ = json.loads(sys.argv[3].replace("\'", "\""))
    which_example = sys.argv[4]


# -------------------Load Model---------------------#

def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir, switch=1):
    '''load model and optimizer'''

    checkpoint = torch.load(save_dir, map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    if switch:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('preobtimizer loaded!')

    print('Pretrained model loaded!')

    return model, optimizer, scheduler


# ----------------------------------------------------------

device = torch.device('cpu')

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

res = 64
data_path_input = '../../Dataset/Wave/64_64/u0s_K24.npy'
data_path_output = '../../Dataset/Wave/64_64/uTs_K24.npy'
pre_model_save_path = folder + "/CN0_d6_checkpoint1000.pt"

if which_example == "shear_layer":
    example = ShearLayer(model_architecture_, device, batch_size, training_samples)
elif which_example == "poisson":
    example = SinFrequency(model_architecture_, device, batch_size, training_samples)
elif which_example == "wave_0_5":
    example = WaveEquation(model_architecture_, device, batch_size, res, data_path_input, data_path_output, training_samples, 64)
elif which_example == "allen":
    example = AllenCahn(model_architecture_, device, batch_size, training_samples)
elif which_example == "cont_tran":
    example = ContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "disc_tran":
    example = DiscContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "airfoil":
    example = Airfoil(model_architecture_, device, batch_size, training_samples)
elif which_example == "Heatdiffusion":
    example = Heatdiffusion(model_architecture_, device, batch_size, res, data_path_input, data_path_output, training_samples, 64)
else:
    raise ValueError()

# -----------------------------------Train--------------------------------------

model = example.model
training_set = example.train_loader #TRAIN LOADER
testing_set = example.val_loader #VALIDATION LOADER

min_vals_input = example.min_vals_input
max_vals_input = example.max_vals_input
min_vals_output = example.min_vals_output
max_vals_output = example.max_vals_output

n_train =512
# ---------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()


if pre_model_save_path:
    fno, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, pre_model_save_path, switch=None)

for param_group in optimizer.param_groups:
    print(param_group['lr'])


test_relative_l2 = 0.0
for step, (input_batch, output_batch) in enumerate(testing_set):

    input_batch = input_batch.to(device)
    output_batch = output_batch.to(device)
    output_pred_batch = model(input_batch)

    output_pred_batch = output_pred_batch * (max_vals_output - min_vals_output) + min_vals_output
    output_batch = output_batch * (max_vals_output - min_vals_output) + min_vals_output

    if which_example == "airfoil":  # Mask the airfoil shape
        output_pred_batch[input_batch == 1] = 1
        output_batch[input_batch == 1] = 1

    loss_f = (torch.mean((output_pred_batch - output_batch) ** 2) / torch.mean(output_batch ** 2)) ** 0.5 * 100
    test_relative_l2 += loss_f.item()
test_relative_l2 /= len(testing_set)

print(test_relative_l2)

print(output_batch.shape)
print(output_pred_batch.shape)
res = 64
x = torch.linspace(0, 1, res)
y = torch.linspace(0, 1, res)
x_grid, y_grid = torch.meshgrid(x, y)


fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
im1 = axs[0].contourf(x_grid,y_grid,output_batch[23,0,:,:],levels = 64,cmap="jet")
#im1 = axs[0].imshow(output_batch[23, 0, :, :], cmap="jet")
axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")
plt.colorbar(im1, ax=axs[0])
axs[0].grid(True, which="both", ls=":")
axs[0].set_title("true_figure")
axs[0].set(aspect='equal')
im2 = axs[1].contourf(x_grid,y_grid,output_pred_batch.detach()[23,0,:,:],levels = 64,cmap="jet")
#im2 = axs[1].imshow(output_pred_batch.detach()[23, 0, :, :], cmap="jet")
axs[1].set_xlabel("x1")
axs[1].set_ylabel("x2")
plt.colorbar(im2, ax=axs[1])
axs[1].grid(True, which="both", ls=":")
axs[1].set_title("prediction")
axs[1].set(aspect='equal')
plt.savefig('../assets/CNO_K4.png', dpi=150)