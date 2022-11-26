"""
*Preliminary* pytorch implementation.

VoxelMorph training.
"""


# python imports
import os
import glob
import random
import warnings
from argparse import ArgumentParser

# external imports
import numpy as np
import torch
from torch.optim import Adam

# internal imports
from model import SPnet
import datagenerators
import losses
from losses import ncc_loss ,gradient_loss


def train(gpu,
          data_dir,
          lr,
          n_iter,
          data_loss,
          model,
          batch_size,
          n_save_iter,
          model_dir):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Produce the loaded atlas with dims.:160x192x224.
    #atlas_vol = np.load(atlas_file)['vol'][np.newaxis, ..., np.newaxis]
    vol_size = [160,192,160]
    train_vol_names = glob.glob(os.path.join(data_dir, '*.npy'))
    # Get all the names of the training data
    
    random.shuffle(train_vol_names)

    model = SPnet(vol_size)
    model.to(device)

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    gen = datagenerators.example_gen(train_vol_names, batch_size)
    pair_gen = datagenerators.cvpr2018_gen_s2s(gen,batch_size)

    for i in range(0,n_iter):

        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)

        # Generate the moving images and convert them to tensors.
        moving_image,fixed_image = next(pair_gen)
        input_moving = torch.from_numpy(moving_image).to(device).float()
        input_moving = input_moving.permute(0, 4, 1, 2, 3)
        input_fixed = torch.from_numpy(fixed_image).to(device).float()
        input_fixed = input_fixed.permute(0, 4, 1, 2, 3)

        
        ([wraped,wraped1,wraped2,wraped3],[fixed,fixed1,fixed2,fixed3],
        [sym_wraped,sym_wraped1,sym_wraped2,sym_wraped3],[moving,moving1,moving2,moving3],
        [flow0,flow1,flow2,flow3],[sym_flow0,sym_flow1,sym_flow2,sym_flow3],
        [vec0,vec1,vec2,vec3]) = model(input_fixed,input_moving)
        
        sim_loss0,sim_loss1,sim_loss2,sim_loss3 = (ncc_loss(fixed,wraped),ncc_loss(fixed1,wraped1,[7,7,7]),
                                                    ncc_loss(fixed2,wraped2,[5,5,5]), ncc_loss(fixed3,wraped3,[3,3,3]))
        sym_sim_loss0,sym_sim_loss1,sym_sim_loss2,sym_sim_loss3 = (ncc_loss(moving,sym_wraped),ncc_loss(moving1,sym_wraped1,[7,7,7]),
                                                    ncc_loss(moving2,sym_wraped2,[5,5,5]), ncc_loss(moving3,sym_wraped3,[3,3,3]))
        grad_loss0,grad_loss1,grad_loss2,grad_loss3 = (gradient_loss(vec0),gradient_loss(vec1),
                                                        gradient_loss(vec2),gradient_loss(vec3))

        loss0 = (sim_loss0 + sym_sim_loss0)/2.0 + grad_loss0 
        loss1 = (sim_loss1 + sym_sim_loss1)/2.0 + grad_loss1
        loss2 = (sim_loss2 + sym_sim_loss2)/2.0 + grad_loss2
        loss3 = (sim_loss3 + sym_sim_loss3)/2.0 + grad_loss3

        loss = loss0 + 0.5*loss1 + 0.25*loss2 + 0.125*loss3

        print("%d,%f,%f,%f,%f" % (i, loss0.item(), sim_loss0.item(),sym_sim_loss0.item(), grad_loss0.item()), flush=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
    save_file_name = os.path.join(model_dir, 'final_model.ckpt' )
    torch.save(model.state_dict(), save_file_name)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='1',
                        help="gpu id")

    parser.add_argument("--data_dir",
                        type=str,
                        default= './data/train',
                        help="data folder with training vols")

    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=1e-4,
                        help="learning rate")

    parser.add_argument("--n_iter",
                        type=int,
                        dest="n_iter",
                        default=30000,
                        help="number of iterations")

    parser.add_argument("--data_loss",
                        type=str,
                        dest="data_loss",
                        default='ncc',
                        help="data_loss: mse of ncc")

    parser.add_argument("--batch_size", 
                        type=int,
                        dest="batch_size", 
                        default=1,
                        help="batch_size")

    parser.add_argument("--n_save_iter", 
                        type=int,
                        dest="n_save_iter", 
                        default=1000,
                        help="frequency of model saves")

    parser.add_argument("--model_dir", 
                        type=str,
                        dest="model_dir", 
                        default='./model/',
                        help="models folder")


    train(**vars(parser.parse_args()))

