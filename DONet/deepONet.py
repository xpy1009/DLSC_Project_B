'''
Usage:
python DONet/deepONet.py -e diffusion
python DONet/deepONet.py -e wave

Trained on MacBook Air M2 within 30 min 
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import deepxde as dde
import argparse

parser = argparse.ArgumentParser(description='For choosing the equation to solve.')
parser.add_argument('-e')

args = parser.parse_args()

if (args.e == 'diffusion'): 
  n_train =150
  n_test = 50
  n_d = 6
  n_res = 4
elif (args.e == 'wave'):
  n_train = 512
  n_test = 128
  n_d = 1
  n_res = 1
else:
  print(args.e + " not implemented")
  exit(0)

losses = np.empty([n_res, n_d])

for d in range(n_d):
  np.random.seed(0)
  torch.manual_seed(0)
  if (args.e == 'diffusion'): 
    u_0 = np.load('Dataset/' + args.e + '/64_64/u0s_d' + str(d+1) + '.npy')
    u_T = np.load('Dataset/' + args.e + '/64_64/uTs_d' + str(d+1) + '.npy')
  else:
    u_0 = np.load('Dataset/' + args.e + '/64_64/u0s_K24.npy')
    u_T = np.load('Dataset/' + args.e + '/64_64/uTs_K24.npy')

  # print(u_0.shape)
  # print(u_T.shape)
  # assert(0)

  input_x = u_0[0, :, :, 1:].reshape(-1, 2)

  # normalize data
  mean_in = u_0[:n_train,:,:,0].mean()
  std_in = u_0[:n_train,:,:,0].std()
  mean_out = u_T[:n_train,:,:].mean()
  std_out = u_T[:n_train,:,:].std()

  tr_inputs = (u_0[:n_train,:,:,0].reshape(n_train, -1) - mean_in) / std_in
  tr_label = (u_T[:n_train,:,:].reshape(n_train, -1) - mean_out) / std_out
  ts_inputs = (u_0[n_train:,:,:,0].reshape(n_test, -1) - mean_in) / std_in
  ts_label = (u_T[n_train:,:,:].reshape(n_test, -1) - mean_out) / std_out

  # print(tr_inputs.shape, tr_label.shape)
  # assert(0)

  # build dataset
  X_train = (tr_inputs.astype(np.float32), input_x.astype(np.float32))
  y_train = tr_label.astype(np.float32)
  X_test = (ts_inputs.astype(np.float32), input_x.astype(np.float32))
  y_test = ts_label.astype(np.float32)
  data = dde.data.TripleCartesianProd(
      X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
  )

  # print(X_test[0].shape, X_test[1].shape, y_test.shape)
  # assert(0)

  # choose network
  m = input_x.shape[0]
  dim_x = 2
  if (n_d>1):
    layers = 40
  else:
    layers = 80
  net = dde.nn.DeepONetCartesianProd(
      [m, layers, layers],
      [dim_x, layers, layers],
      "relu",
      "Glorot normal",
  )
  # print(net)
  # assert(0)

  # Define a Model
  model = dde.Model(data, net)

  # Compile and train.
  model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
  losshistory, train_state = model.train(iterations=10000)

  # Plot the loss trajectory
  # dde.utils.plot_loss_history(losshistory)
  # plt.savefig('assets/DON/' + args.e + '/loss_d' + str(d+1) + '.png', dpi=150)

  # evaluate model
  for r in range(n_res):
    if (args.e == 'diffusion'):
      res = 32 * (r+1)

      u_0 = np.load('Dataset/' + args.e + '/' + str(res) + '_' + str(res) + '/u0s_d' + str(d+1) + '.npy')
      u_T = np.load('Dataset/' + args.e + '/' + str(res) + '_' + str(res) + '/uTs_d' + str(d+1) + '.npy')
      if (r==0):
        ev_inputs = (u_0[n_train:,:,:,0].repeat(2,axis=1).repeat(2,axis=2).reshape(n_test, -1) - mean_in) / std_in
      else:
        idx = np.round(np.linspace(0, res - 1, 64)).astype(int)
        tmp = u_0[n_train:,:,:,0]
        tmp = tmp[:,:,idx]
        tmp = tmp[:,idx,:]
        ev_inputs = (tmp.reshape(n_test, -1) - mean_in) / std_in

      ev_label = (u_T[n_train:,:,:].reshape(n_test, -1) - mean_out) / std_out
      ev_loc = u_0[0, :, :, 1:].reshape(-1, 2)

      X_eval = (ev_inputs.astype(np.float32), ev_loc.astype(np.float32))
      y_eval = ev_label.astype(np.float32)
    else:
      X_eval = X_test
      y_eval = y_test
      ev_loc = input_x

    pred = model.predict(X_eval)
    # print(pred.shape)
    pred = pred * std_out + mean_out
    gt = y_eval * std_out + mean_out
    err = (np.mean((pred - gt)**2) / np.mean(gt**2))**0.5 * 100
    losses[r, d] = err

  # visualize results
  # fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
  # im1 = axs[0].scatter(ev_loc[:, 1], ev_loc[:, 0], c=pred[0,:], cmap="jet")
  # axs[0].set_xlabel("x1")
  # axs[0].set_ylabel("x2")
  # plt.colorbar(im1, ax=axs[0])
  # axs[0].grid(True, which="both", ls=":")
  # axs[0].set_title("pred")
  # im2 = axs[1].scatter(ev_loc[:, 1], ev_loc[:, 0], c=gt[0,:], cmap="jet")
  # axs[1].set_xlabel("x1")
  # axs[1].set_ylabel("x2")
  # plt.colorbar(im2, ax=axs[1])
  # axs[1].grid(True, which="both", ls=":")
  # axs[1].set_title("ground truth")
  # plt.show()
  # plt.savefig('assets/DON/' + args.e + '/res_d' + str(d+1) + '.png', dpi=150)

print("MSE Losses:", losses)
