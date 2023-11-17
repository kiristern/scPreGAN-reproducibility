from __future__ import print_function
import os
from pathlib import Path
from functools import partial
import random
import json
import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.cuda import is_available as cuda_is_available
from torch import Tensor, FloatTensor
from torch.utils.data import random_split
from torch import autograd
from torch import mean, exp, unique, cat, isnan
from torch import norm as torch_norm
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import pandas as pd


import torch.nn.functional as F
import scanpy as sc
import anndata
from scipy import sparse

from ray import tune
from ray.tune import CLIReporter

from util import load_anndata,label_encoder 

from model.Discriminator import Discriminator_AC
from model.Generator import Generator_AC_layer
from model.Encoder import Encoder_AC_layer

import warnings

warnings.filterwarnings('ignore')


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight, 1e-2)
        m.bias.data.fill_(0.01)


def create_model(n_features, z_dim, min_hidden_size, n_classes, use_cuda, use_sn, train_flag):
    D_A = Discriminator_AC(n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1, n_classes=n_classes)
    D_B = Discriminator_AC(n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1, n_classes=n_classes)

    G_A = Generator_AC_layer(z_dim=z_dim, min_hidden_size=min_hidden_size, n_features=n_features, n_classes=n_classes)
    G_B = Generator_AC_layer(z_dim=z_dim, min_hidden_size=min_hidden_size, n_features=n_features, n_classes=n_classes)
    E = Encoder_AC_layer(n_features=n_features, min_hidden_size=min_hidden_size, z_dim=z_dim)
    if not train_flag:
        return E, G_A, G_B, D_A, D_B

    # print("Encoder model:")
    # print(E)
    init_weights(E)
    # print("GeneratorA model:")
    # print(G_A)
    init_weights(G_A)
    init_weights(G_B)
    # print("disc_model:")
    # print(D_A)
    init_weights(D_A)
    init_weights(D_B)

    if use_cuda and torch.cuda.is_available():
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        G_A = G_A.cuda()
        G_B = G_B.cuda()
        E = E.cuda()

    return E, G_A, G_B, D_A, D_B

def train_scPreGAN(config, opt):
    if opt['manual_seed'] is None:
        opt['manual_seed'] = random.randint(1, 10000)
    print("Random Seed: ", opt['manual_seed'])
    random.seed(opt['manual_seed'])
    torch.manual_seed(opt['manual_seed'])
    if opt['cuda']:
        torch.cuda.manual_seed_all(opt['manual_seed'])
    A_pd, A_celltype_ohe_pd, B_pd, B_celltype_ohe_pd = load_anndata(path=opt['dataPath'],
                                                                    condition_key=opt['condition_key'],
                                                                    condition=opt['condition'],
                                                                    cell_type_key=opt['cell_type_key'],
                                                                    prediction_type=opt['prediction_type'],
                                                                    out_sample_prediction=opt['out_sample_prediction']
                                                                    )
    trainA = [np.array(A_pd), np.array(A_celltype_ohe_pd)]  
    trainB = [np.array(B_pd), np.array(B_celltype_ohe_pd)]

    expr_trainA, cell_type_trainA = trainA
    expr_trainB, cell_type_trainB = trainB

    expr_trainA_tensor = Tensor(expr_trainA)
    expr_trainB_tensor = Tensor(expr_trainB)
    cell_type_trainA_tensor = Tensor(cell_type_trainA)
    cell_type_trainB_tensor = Tensor(cell_type_trainB)


    if opt['cuda'] and torch.cuda.is_available():
        # A_tensor = A_tensor.cuda()
        # B_tensor = B_tensor.cuda()
        expr_trainA_tensor = expr_trainA_tensor.cuda()
        expr_trainB_tensor = expr_trainB_tensor.cuda()
        cell_type_trainA_tensor = cell_type_trainA_tensor.cuda()
        cell_type_trainB_tensor = cell_type_trainB_tensor.cuda()

    A_Dataset = torch.utils.data.TensorDataset(expr_trainA_tensor, cell_type_trainA_tensor)
    B_Dataset = torch.utils.data.TensorDataset(expr_trainB_tensor, cell_type_trainB_tensor)


    if opt['validation'] and opt['valid_dataPath'] is None:
        print('splite dataset to train subset and validation subset')
        A_test_abs = int(len(A_Dataset) * 0.8)
        A_train_subset, A_val_subset = random_split(
            A_Dataset, [A_test_abs, len(A_Dataset) - A_test_abs])

        B_test_abs = int(len(B_Dataset) * 0.8)
        B_train_subset, B_val_subset = random_split(
            B_Dataset, [B_test_abs, len(B_Dataset) - B_test_abs])

        A_train_loader = torch.utils.data.DataLoader(dataset=A_train_subset,
                                                     batch_size=int(config['batch_size']),
                                                     shuffle=True,
                                                     drop_last=True)

        B_train_loader = torch.utils.data.DataLoader(dataset=B_train_subset,
                                                     batch_size=int(config['batch_size']),
                                                     shuffle=True,
                                                     drop_last=True)
        A_valid_loader = torch.utils.data.DataLoader(dataset=A_val_subset,
                                                     batch_size=int(config['batch_size']),
                                                     shuffle=True,
                                                     drop_last=True)
        B_valid_loader = torch.utils.data.DataLoader(dataset=B_val_subset,
                                                     batch_size=int(config['batch_size']),
                                                     shuffle=True,
                                                     drop_last=True)
    elif opt['validation'] and opt['valid_dataPath'] is not None:
        A_pd_val, A_celltype_ohe_pd_val, B_pd_val, B_celltype_ohe_pd_val = load_anndata(path=opt['valid_dataPath'],
                                                                                        condition_key=opt[
                                                                                            'condition_key'],
                                                                                        condition=opt['condition'],
                                                                                        cell_type_key=opt[
                                                                                            'cell_type_key'])

        print(f"use validation dataset, lenth of A: {A_pd_val.shape}, lenth of B: {B_pd_val.shape}")

        valA = [np.array(A_pd), np.array(A_celltype_ohe_pd)]
        valB = [np.array(B_pd), np.array(B_celltype_ohe_pd)]

        expr_valA, cell_type_valA = valA
        expr_valB, cell_type_valB = valB

        expr_valA_tensor = Tensor(expr_valA)
        expr_valB_tensor = Tensor(expr_valB)
        cell_type_valA_tensor = Tensor(cell_type_valA)
        cell_type_valB_tensor = Tensor(cell_type_valB)

        if opt['cuda'] and torch.cuda.is_available():
            expr_valA_tensor = expr_valA_tensor.cuda()
            expr_valB_tensor = expr_valB_tensor.cuda()
            cell_type_valA_tensor = cell_type_valA_tensor.cuda()
            cell_type_valB_tensor = cell_type_valB_tensor.cuda()

        A_Dataset_val = torch.utils.data.TensorDataset(expr_valA_tensor, cell_type_valA_tensor)
        B_Dataset_val = torch.utils.data.TensorDataset(expr_valB_tensor, cell_type_valB_tensor)

        A_train_loader = torch.utils.data.DataLoader(dataset=A_Dataset,
                                                     batch_size=int(config['batch_size']),
                                                     shuffle=True,
                                                     drop_last=True)
        B_train_loader = torch.utils.data.DataLoader(dataset=B_Dataset,
                                                     batch_size=int(config['batch_size']),
                                                     shuffle=True,
                                                     drop_last=True)
        A_valid_loader = torch.utils.data.DataLoader(dataset=A_Dataset_val,
                                                     batch_size=int(config['batch_size']),
                                                     shuffle=True,
                                                     drop_last=True)
        B_valid_loader = torch.utils.data.DataLoader(dataset=B_Dataset_val,
                                                     batch_size=int(config['batch_size']),
                                                     shuffle=True,
                                                     drop_last=True)
    else:
        print('No validation.')
        A_train_loader = torch.utils.data.DataLoader(dataset=A_Dataset,
                                                     batch_size=int(config['batch_size']),
                                                     shuffle=True,
                                                     drop_last=True)

        B_train_loader = torch.utils.data.DataLoader(dataset=B_Dataset,
                                                     batch_size=int(config['batch_size']),
                                                     shuffle=True,
                                                     drop_last=True)



    opt['n_features'] = A_pd.shape[1]
    n_classes = opt['n_classes']
    print("feature length: ", opt['n_features'])
    A_train_loader_it = iter(A_train_loader)
    B_train_loader_it = iter(B_train_loader)

    E, G_A, G_B, D_A, D_B = create_model(n_features=opt['n_features'],
                                         z_dim=config['z_dim'],
                                         min_hidden_size=config['min_hidden_size'],
                                         use_cuda=opt['cuda'], use_sn=opt['use_sn'], 
                                         n_classes=n_classes, train_flag=opt['train_flag'])

    recon_criterion = nn.MSELoss()
    encoding_criterion = nn.MSELoss()
    dis_criterion = nn.BCELoss()
    aux_criterion = nn.NLLLoss()

    optimizerD_A = torch.optim.Adam(D_A.parameters(), lr=config['lr_disc'], betas=(0.5, 0.9))
    optimizerD_B = torch.optim.Adam(D_B.parameters(), lr=config['lr_disc'], betas=(0.5, 0.9))
    optimizerG_A = torch.optim.Adam(G_A.parameters(), lr=config['lr_g'], betas=(0.5, 0.9))
    optimizerG_B = torch.optim.Adam(G_B.parameters(), lr=config['lr_g'], betas=(0.5, 0.9))
    optimizerE = torch.optim.Adam(E.parameters(), lr=config['lr_e'])

    ones = torch.ones(config['batch_size'], 1)
    print('ones type:', type(ones))
    zeros = torch.zeros(config['batch_size'], 1)

    if opt['cuda'] and cuda_is_available():
        ones = ones.cuda()
        zeros = zeros.cuda()

    D_A.train()
    D_B.train()
    G_A.train()
    G_B.train()
    E.train()

    D_A_loss = 0.0
    D_B_loss = 0.0

    iteration = 0
    for iteration in range(1, config['niter'] + 1):
        if iteration % 10000 == 0:
            for param_group in optimizerD_A.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
            for param_group in optimizerD_B.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
            for param_group in optimizerG_A.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
            for param_group in optimizerG_B.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
            for param_group in optimizerE.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9


        try:
            real_A, cell_type_A = next(A_train_loader_it)
            real_B, cell_type_B = next(B_train_loader_it)
        except StopIteration:
            A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(B_train_loader)
            real_A, cell_type_A = next(A_train_loader_it)
            real_B, cell_type_B = next(B_train_loader_it)

        if (opt['cuda']) and cuda_is_available():
            real_A = real_A.cuda()
            real_B = real_B.cuda()
            cell_type_A = cell_type_A.cuda()
            cell_type_B = cell_type_B.cuda()

        D_A.zero_grad()
        D_B.zero_grad()

        out_A, out_A_cls = D_A(real_A, cell_type_A) # real A
        out_B, out_B_cls = D_B(real_B, cell_type_B) # real B

        real_A_z = E(real_A)
        AB = G_B(real_A_z)
        
        real_B_z = E(real_B)
        BA = G_A(real_B_z)

        out_BA, out_BA_cls = D_A(BA.detach(), cell_type_B) # false A
        out_AB, out_AB_cls = D_B(AB.detach(), cell_type_A) # false B

        _cell_type_A = torch.argmax(cell_type_A, dim=-1)
        _cell_type_B = torch.argmax(cell_type_B, dim=-1)

        dis_D_A_real = dis_criterion(out_A, ones)
        aux_D_A_real = aux_criterion(out_A_cls, _cell_type_A)
        D_A_real = dis_D_A_real + aux_D_A_real
        dis_D_A_fake = dis_criterion(out_BA, zeros)
        aux_D_A_fake = aux_criterion(out_BA_cls, _cell_type_B) 
        D_A_fake = dis_D_A_fake + aux_D_A_fake

        dis_D_B_real = dis_criterion(out_B, ones)
        aux_D_B_real = aux_criterion(out_B_cls, _cell_type_B)
        D_B_real = dis_D_B_real + aux_D_B_real
        dis_D_B_fake = dis_criterion(out_AB, zeros)
        aux_D_B_fake = aux_criterion(out_AB_cls, _cell_type_A)
        D_B_fake = dis_D_B_fake + aux_D_B_fake

        D_A_loss = D_A_real + D_A_fake
        D_B_loss = D_B_real + D_B_fake

        D_A_loss.backward()
        D_B_loss.backward()
        optimizerD_A.step()
        optimizerD_B.step()     

        try:
            real_A, cell_type_A = next(A_train_loader_it)
            real_B, cell_type_B = next(B_train_loader_it)
        except StopIteration:
            A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(B_train_loader)
            real_A, cell_type_A = next(A_train_loader_it)
            real_B, cell_type_B = next(B_train_loader_it)

        if (opt['cuda']) and cuda_is_available():
            real_A = real_A.cuda()
            real_B = real_B.cuda()
            cell_type_A = cell_type_A.cuda()
            cell_type_B = cell_type_B.cuda()

        G_A.zero_grad()
        G_B.zero_grad()
        E.zero_grad()

        real_A_z = E(real_A)
        AA = G_A(real_A_z)
        AB = G_B(real_A_z)

        AA_z = E(AA)
        AB_z = E(AB)
        ABA = G_A(AB_z)

        real_B_z = E(real_B)
        BA = G_A(real_B_z)
        BB = G_B(real_B_z)
        BA_z = E(BA)
        BB_z = E(BB)
        BAB = G_B(BA_z)

        out_AA, out_AA_cls = D_A(AA, cell_type_A)
        out_AB, out_AB_cls = D_B(AB, cell_type_A)
        out_BA, out_BA_cls = D_A(BA, cell_type_B)
        out_BB, out_BB_cls = D_B(BB, cell_type_B)
        out_ABA, out_ABA_cls = D_A(ABA, cell_type_A)
        out_BAB, out_BAB_cls = D_B(BAB, cell_type_B)

        G_AA_adv_loss = dis_criterion(out_AA, ones) + aux_criterion(out_AA_cls, _cell_type_A)
        G_BA_adv_loss = dis_criterion(out_BA, ones) + aux_criterion(out_BA_cls, _cell_type_B)
        G_ABA_adv_loss = dis_criterion(out_ABA, ones) + aux_criterion(out_ABA_cls, _cell_type_A)
    
        G_BB_adv_loss = dis_criterion(out_BB, ones) + aux_criterion(out_BB_cls, _cell_type_B)
        G_AB_adv_loss = dis_criterion(out_AB, ones) + aux_criterion(out_AB_cls, _cell_type_A)
        G_BAB_adv_loss = dis_criterion(out_BAB, ones) + aux_criterion(out_BAB_cls, _cell_type_B)
    
        G_A_adv_loss = G_AA_adv_loss + G_BA_adv_loss + G_ABA_adv_loss
        G_B_adv_loss = G_BB_adv_loss + G_AB_adv_loss + G_BAB_adv_loss

        adv_loss = (G_A_adv_loss + G_B_adv_loss) * config['lambda_adv']

        # reconstruction loss
        l_rec_AA = recon_criterion(AA, real_A)
        l_rec_BB = recon_criterion(BB, real_B)

        recon_loss = (l_rec_AA + l_rec_BB) * config['lambda_recon']

        # encoding loss
        tmp_real_A_z = real_A_z.detach()
        tmp_real_B_z = real_B_z.detach()
        l_encoding_AA = encoding_criterion(AA_z, tmp_real_A_z)
        l_encoding_BB = encoding_criterion(BB_z, tmp_real_B_z)
        l_encoding_BA = encoding_criterion(BA_z, tmp_real_B_z)
        l_encoding_AB = encoding_criterion(AB_z, tmp_real_A_z)

        encoding_loss = (l_encoding_AA + l_encoding_BB + l_encoding_BA + l_encoding_AB) * config[
            'lambda_encoding']

        G_loss = adv_loss + recon_loss + encoding_loss

        # backward
        G_loss.backward()

        # step
        optimizerG_A.step()
        optimizerG_B.step()
        optimizerE.step()

        tune.report(D_A_loss=(D_A_loss.item()),
                    D_B_loss=(D_B_loss.item()),
                    adv_loss=(adv_loss.item()),
                    recon_loss=(recon_loss.item()),
                    encoding_loss=(encoding_loss.item()),
                    G_loss=(G_loss.item())
                    )

        if iteration % 300 == 0:
            print(
                '[%d/%d] D_A_loss: %.4f  D_B_loss: %.4f adv_loss: %.4f  recon_loss: %.4f encoding_loss: %.4f G_loss: %.4f'
                % (iteration, config['niter'], D_A_loss.item(), D_B_loss.item(), adv_loss.item(), recon_loss.item(),
                   encoding_loss.item(), G_loss.item()))
            D_A_loss_val = 0.0
            D_B_loss_val = 0.0
            adv_loss_val = 0.0
            recon_loss_val = 0.0
            encoding_loss_val = 0.0
            G_loss_val = 0.0

            counter = 0

            A_valid_loader_it = iter(A_valid_loader)
            B_valid_loader_it = iter(B_valid_loader)

            max_length = max(len(A_valid_loader), len(B_valid_loader))
            with torch.no_grad():
                for iteration_val in range(1, max_length):
                    try:
                        cellA_val, cellA_val_type = next(A_valid_loader_it)
                        cellB_val, cellB_val_type = next(B_valid_loader_it)
                    except StopIteration:
                        A_valid_loader_it, B_valid_loader_it = iter(A_valid_loader), iter(B_valid_loader)
                        cellA_val, cellA_val_type = next(A_valid_loader_it)
                        cellB_val, cellB_val_type = next(B_valid_loader_it)


                    counter += 1

                    real_A_z = E(cellA_val)
                    real_B_z = E(cellB_val)
                    AB = G_B(real_A_z)
                    BA = G_A(real_B_z)
                    AA = G_A(real_A_z)
                    BB = G_B(real_B_z)
                    AA_z = E(AA)
                    BB_z = E(BB)
                    AB_z = E(AB)
                    BA_z = E(BA)

                    ABA = G_A(AB_z)
                    BAB = G_B(BA_z)

                    outA_val, outA_val_cls = D_A(cellA_val, cellA_val_type)
                    outB_val, outB_val_cls = D_B(cellB_val, cellB_val_type)
                    out_AA, out_AA_cls = D_A(AA, cellA_val_type)
                    out_BB, out_BB_cls = D_B(BB, cellB_val_type)
                    out_AB, out_AB_cls = D_B(AB, cellA_val_type)
                    out_BA, out_BA_cls = D_A(BA, cellB_val_type)
                    out_ABA, out_ABA_cls = D_A(ABA, cellA_val_type)
                    out_BAB, out_BAB_cls = D_B(BAB, cellB_val_type)

                    _cell_type_A = torch.argmax(cellA_val_type, dim=-1)
                    _cell_type_B = torch.argmax(cellB_val_type, dim=-1)

                    dis_D_A_real = dis_criterion(outA_val, ones)
                    aux_D_A_real = aux_criterion(outA_val_cls, _cell_type_A)
                    D_A_real_loss_val = dis_D_A_real + aux_D_A_real
                    dis_D_A_fake = dis_criterion(out_BA, zeros) 
                    aux_D_A_fake = aux_criterion(out_BA_cls, _cell_type_B) 
                    D_A_fake_loss_val = dis_D_A_fake + aux_D_A_fake

                    dis_D_B_real = dis_criterion(outB_val, ones)
                    aux_D_B_real = aux_criterion(outB_val_cls, _cell_type_B)
                    D_B_real_loss_val = dis_D_B_real + aux_D_B_real
                    dis_D_B_fake = dis_criterion(out_AB, zeros) 
                    aux_D_B_fake = aux_criterion(out_AB_cls, _cell_type_A) 
                    D_B_fake_loss_val = dis_D_B_fake + aux_D_B_fake

                    D_A_loss_val += (D_A_real_loss_val + D_A_fake_loss_val).item()
                    D_B_loss_val += (D_B_real_loss_val + D_B_fake_loss_val).item()

                    G_AA_adv_loss_val = dis_criterion(out_AA, ones) + aux_criterion(out_AA_cls, _cell_type_A)
                    G_BA_adv_loss_val = dis_criterion(out_BA, ones) + aux_criterion(out_BA_cls, _cell_type_B)
                    G_ABA_adv_loss_val = dis_criterion(out_ABA, ones) + aux_criterion(out_ABA_cls, _cell_type_A)
                
                    G_BB_adv_loss_val = dis_criterion(out_BB, ones) + aux_criterion(out_BB_cls, _cell_type_B)
                    G_AB_adv_loss_val = dis_criterion(out_AB, ones) + aux_criterion(out_AB_cls, _cell_type_A)
                    G_BAB_adv_loss_val = dis_criterion(out_BAB, ones) + aux_criterion(out_BAB_cls, _cell_type_B)

                    G_A_adv_loss_val = G_AA_adv_loss_val + G_BA_adv_loss_val + G_ABA_adv_loss_val
                    G_B_adv_loss_val = G_BB_adv_loss_val + G_AB_adv_loss_val + G_BAB_adv_loss_val
                    adv_loss_val += (G_A_adv_loss_val + G_B_adv_loss_val).item() * config['lambda_adv']

                    # reconstruction loss
                    l_rec_AA_val = recon_criterion(AA, cellA_val)
                    l_rec_BB_val = recon_criterion(BB, cellB_val)
                    recon_loss_val += (l_rec_AA_val + l_rec_BB_val).item() * config['lambda_recon']

                    # encoding loss
                    l_encoding_AA_val = encoding_criterion(AA_z, real_A_z)
                    l_encoding_BB_val = encoding_criterion(BB_z, real_B_z)
                    l_encoding_BA_val = encoding_criterion(BA_z, real_B_z)
                    l_encoding_AB_val = encoding_criterion(AB_z, real_A_z)
                    encoding_loss_val += (
                                                 l_encoding_AA_val + l_encoding_BB_val + l_encoding_BA_val + l_encoding_AB_val).item() * \
                                         config['lambda_encoding']
                    G_loss_val += adv_loss_val + recon_loss_val + encoding_loss_val

            print(
                '[%d/%d] adv_loss_val: %.4f  recon_loss_val: %.4f encoding_loss_val: %.4f  G_loss: %.4f D_A_loss_val: %.4f D_B_loss_val: %.4f'
                % (iteration, config['niter'], adv_loss_val / counter, recon_loss_val / counter,
                   encoding_loss_val / counter, G_loss / counter, D_A_loss_val / counter, D_B_loss_val / counter))

            tune.report(adv_loss_val=(adv_loss_val / counter),
                        recon_loss_val=(recon_loss_val / counter),
                        encoding_loss_val=(encoding_loss_val / counter),
                        G_loss=(G_loss / counter),
                        D_A_loss_val=(D_A_loss_val / counter),
                        D_B_loss_val=(D_B_loss_val / counter))

    with tune.checkpoint_dir(iteration) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, f"checkpoint_E_{opt['prediction_type']}")
        torch.save((E.state_dict(), optimizerE.state_dict()), path)
    with tune.checkpoint_dir(iteration) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, f"checkpoint_G_A_{opt['prediction_type']}")
        torch.save((G_A.state_dict(), optimizerG_A.state_dict()), path)
    with tune.checkpoint_dir(iteration) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, f"checkpoint_G_B_{opt['prediction_type']}")
        torch.save((G_B.state_dict(), optimizerG_B.state_dict()), path)
    with tune.checkpoint_dir(iteration) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, f"checkpoint_D_A_{opt['prediction_type']}")
        torch.save((D_A.state_dict(), optimizerD_A.state_dict()), path)
    with tune.checkpoint_dir(iteration) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, f"checkpoint_D_B_{opt['prediction_type']}")
        torch.save((D_B.state_dict(), optimizerD_B.state_dict()), path)
    print("Finished Training")


def test_results(opt, num_trial, trial_i):
    print("+++++++++++++++++++++trial " + str(num_trial) + "+++++++++++++++++++++++++++")
    outf = opt['outf'] + '/trial_' + str(num_trial)
    adata = sc.read(opt['dataPath'])
    n_features = adata.shape[1]
    n_classes = opt['n_classes']

    # 读取模型，得到全部的预测数据和隐向量数据
    cell_type_list = adata.obs[opt['cell_type_key']].unique().tolist()
    for cell_type in cell_type_list:
        opt['prediction_type'] = cell_type
        E, G_A, G_B, D_A, D_B = create_model(n_features=n_features,
                                             z_dim=trial_i.config['z_dim'],
                                             min_hidden_size=trial_i.config['min_hidden_size'],
                                             use_cuda=opt['cuda'], use_sn=opt['use_sn'],n_classes=n_classes, train_flag=opt['train_flag'])

        checkpoint_dir = trial_i.checkpoint.value

        E_state, optimizerE_state = torch.load(os.path.join(checkpoint_dir, f"checkpoint_E_{opt['prediction_type']}"))
        E.load_state_dict(E_state)
        G_A_state, optimizerG_A_state = torch.load(os.path.join(checkpoint_dir, f"checkpoint_G_A_{opt['prediction_type']}"))
        G_A.load_state_dict(G_A_state)
        G_B_state, optimizerG_B_state = torch.load(os.path.join(checkpoint_dir, f"checkpoint_G_B_{opt['prediction_type']}"))
        G_B.load_state_dict(G_B_state)
        D_A_state, optimizerD_A_state = torch.load(os.path.join(checkpoint_dir, f"checkpoint_D_A_{opt['prediction_type']}"))
        D_A.load_state_dict(D_A_state)
        D_B_state, optimizerD_B_state = torch.load(os.path.join(checkpoint_dir, f"checkpoint_D_B_{opt['prediction_type']}"))
        D_B.load_state_dict(D_B_state)


        # 读取相关数据 ############
        adata = sc.read(opt['dataPath'])
        control_adata = adata[(adata.obs[opt['cell_type_key']] == opt['prediction_type']) & (
                adata.obs[opt['condition_key']] == opt['condition']['control'])]
        case_adata = adata[(adata.obs[opt['cell_type_key']] == opt['prediction_type']) & (
                adata.obs[opt['condition_key']] == opt['condition']['case'])]        
        if sparse.issparse(control_adata.X):
            control_pd = pd.DataFrame(data=control_adata.X.A, index=control_adata.obs_names,
                                    columns=control_adata.var_names)
            case_pd = pd.DataFrame(data=case_adata.X.A, index=case_adata.obs_names,
                                    columns=case_adata.var_names)
        else:
            control_pd = pd.DataFrame(data=control_adata.X, index=control_adata.obs_names,
                                    columns=control_adata.var_names)
            case_pd = pd.DataFrame(data=case_adata.X, index=case_adata.obs_names,
                                    columns=case_adata.var_names)
        
        encode_attr = adata.obs[opt['cell_type_key']].unique().tolist()
        adata_celltype_ohe = label_encoder(adata, opt['cell_type_key'], encode_attr)

        adata_celltype_ohe_pd = pd.DataFrame(data=adata_celltype_ohe, index=adata.obs_names)

        control_celltype_ohe_pd = adata_celltype_ohe_pd.loc[control_pd.index, :]
        case_celltype_ohe_pd = adata_celltype_ohe_pd.loc[case_pd.index, :]

        control_data = [np.array(control_pd), np.array(control_celltype_ohe_pd)]
        case_data = [np.array(case_pd), np.array(case_celltype_ohe_pd)]

        expr_control, cell_type_control = control_data
        expr_case, cell_type_case = case_data

        control_tensor = Tensor(expr_control)
        cell_type_control_tensor = Tensor(cell_type_control)
        case_tensor = Tensor(expr_case)
        cell_type_case_tensor = Tensor(cell_type_case)

        ###################


        if opt['cuda'] and cuda_is_available():
            control_tensor = control_tensor.cuda()
            cell_type_control_tensor = cell_type_control_tensor.cuda()
            case_tensor = case_tensor.cuda()
            cell_type_case_tensor = cell_type_case_tensor.cuda()
        control_z = E(control_tensor)
        case_z = E(case_tensor)
        case_pred = G_B(control_z, cell_type_control_tensor)
        real_case_pred = G_B(case_z, cell_type_case_tensor)

        # control的隐向量
        control_z_adata = anndata.AnnData(X=control_z.cpu().detach().numpy(),
                                          obs={opt['condition_key']: ["control_z"] * len(control_z),
                                               opt['cell_type_key']: control_adata.obs[opt['cell_type_key']].tolist()})
        
        save_ctrl_z_path = os.path.join(outf, 'control_z_adata')
        os.makedirs(save_ctrl_z_path, exist_ok=True)
        control_z_adata.write_h5ad(f"{save_ctrl_z_path}/control_z_{opt['prediction_type']}.h5ad")

        # 预测数据
        pred_perturbed_adata = anndata.AnnData(X=case_pred.cpu().detach().numpy(),
                                               obs={opt['condition_key']: ["pred_perturbed"] * len(case_pred),
                                                    opt['cell_type_key']: control_adata.obs[opt['cell_type_key']].tolist()})
        pred_perturbed_adata.var_names = adata.var_names
        
        pred_perturbed_save_path = os.path.join(outf, 'pred_adata')
        os.makedirs(pred_perturbed_save_path, exist_ok=True)
        pred_perturbed_adata.write_h5ad(f"{pred_perturbed_save_path}/pred_{opt['prediction_type']}.h5ad")

        # 预测数据的隐向量
        pred_z = E(case_pred)
        pred_z_adata = anndata.AnnData(X=pred_z.cpu().detach().numpy(),
                                       obs={opt['condition_key']: ["pred_z"] * len(pred_z),
                                            opt['cell_type_key']: pred_perturbed_adata.obs[opt['cell_type_key']].tolist()})
        
        pred_z_save_path = os.path.join(outf, 'pred_z_adata')
        os.makedirs(pred_z_save_path, exist_ok=True)
        pred_z_adata.write_h5ad(f"{pred_z_save_path}/pred_{opt['prediction_type']}.h5ad")

    # 读取数据并组合在一起计算各个指标

    # 读取预测的stimulated数据
    pred_path = outf + "/pred_adata/"
    # B_pred_adata = sc.read(pred_path + "pred_B.h5ad")
    # CD14_Mono_pred_adata = sc.read(pred_path + "pred_CD14+Mono.h5ad")
    # CD4T_pred_adata = sc.read(pred_path + "pred_CD4T.h5ad")
    # CD8T_pred_adata = sc.read(pred_path + "pred_CD8T.h5ad")
    # Dendritic_pred_adata = sc.read(pred_path + "pred_Dendritic.h5ad")
    # FCGR3A_Mono_pred_adata = sc.read(pred_path + "pred_FCGR3A+Mono.h5ad")
    # NK_pred_adata = sc.read(pred_path + "pred_NK.h5ad")
    B_pred_adata = sc.read(pred_path + "pred_CD19+ B.h5ad")
    CD14_Mono_pred_adata = sc.read(pred_path + "pred_CD14+ Monocyte.h5ad")
    Dendritic_pred_adata = sc.read(pred_path + "pred_Dendritic.h5ad")
    NK_pred_adata = sc.read(pred_path + "pred_CD56+ NK.h5ad")

    # pred_adata = B_pred_adata.concatenate(CD14_Mono_pred_adata, CD4T_pred_adata, CD8T_pred_adata, Dendritic_pred_adata,
    #                                       FCGR3A_Mono_pred_adata, NK_pred_adata)
    pred_adata = B_pred_adata.concatenate(CD14_Mono_pred_adata, Dendritic_pred_adata, NK_pred_adata)
    all_adata = adata.concatenate(pred_adata)

    sc.pp.neighbors(pred_adata)
    sc.tl.umap(pred_adata)
    sc.pl.umap(pred_adata, color=[opt['cell_type_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"_pred_{num_trial}_{opt['model_name']}.pdf",
               show=False,
               frameon=False)

    sc.pp.neighbors(all_adata)
    sc.tl.umap(all_adata)
    sc.pl.umap(all_adata, color=[opt['cell_type_key'], opt['condition_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"_all_{num_trial}_{opt['model_name']}.pdf",
               show=False,
               frameon=False)

    # control数据隐向量
    # control_z_path = outf + "/control_z_adata/"
    # B_control_z_adata = sc.read(control_z_path + "control_z_B.h5ad")
    # CD14_Mono_control_z_adata = sc.read(control_z_path + "control_z_CD14+Mono.h5ad")
    # CD4T_control_z_adata = sc.read(control_z_path + "control_z_CD4T.h5ad")
    # CD8T_control_z_adata = sc.read(control_z_path + "control_z_CD8T.h5ad")
    # Dendritic_control_z_adata = sc.read(control_z_path + "control_z_Dendritic.h5ad")
    # FCGR3A_Mono_control_z_adata = sc.read(control_z_path + "control_z_FCGR3A+Mono.h5ad")
    # NK_control_z_adata = sc.read(control_z_path + "control_z_NK.h5ad")
    
    control_z_path = outf + "/control_z_adata/"
    B_control_z_adata = sc.read(control_z_path + "control_z_CD19+ B.h5ad")
    CD14_Mono_control_z_adata = sc.read(control_z_path + "control_z_CD14+ Monocyte.h5ad")
    Dendritic_control_z_adata = sc.read(control_z_path + "control_z_Dendritic.h5ad")
    NK_control_z_adata = sc.read(control_z_path + "control_z_CD56+ NK.h5ad")
    
    # control_z = B_control_z_adata.concatenate(CD14_Mono_control_z_adata, CD4T_control_z_adata, CD8T_control_z_adata,
    #                                           Dendritic_control_z_adata,
    #                                           FCGR3A_Mono_control_z_adata, NK_control_z_adata)

    control_z = B_control_z_adata.concatenate(CD14_Mono_control_z_adata,
                                              Dendritic_control_z_adata,
                                              NK_control_z_adata)
    
    # 预测数据隐向量
    pred_z_path = outf + "/pred_z_adata/"
    # B_pred_z_adata = sc.read(pred_z_path + "pred_z_B.h5ad")
    # CD14_Mono_pred_z_adata = sc.read(pred_z_path + "pred_z_CD14+Mono.h5ad")
    # CD4T_pred_z_adata = sc.read(pred_z_path + "pred_z_CD4T.h5ad")
    # CD8T_pred_z_adata = sc.read(pred_z_path + "pred_z_CD8T.h5ad")
    # Dendritic_pred_z_adata = sc.read(pred_z_path + "pred_z_Dendritic.h5ad")
    # FCGR3A_Mono_pred_z_adata = sc.read(pred_z_path + "pred_z_FCGR3A+Mono.h5ad")
    # NK_pred_z_adata = sc.read(pred_z_path + "pred_z_NK.h5ad")
    B_pred_z_adata = sc.read(pred_z_path + "pred_z_CD19+ B.h5ad")
    CD14_Mono_pred_z_adata = sc.read(pred_z_path + "pred_z_CD14+ Monocyte.h5ad")
    Dendritic_pred_z_adata = sc.read(pred_z_path + "pred_z_Dendritic.h5ad")
    NK_pred_z_adata = sc.read(pred_z_path + "pred_z_CD56+ NK.h5ad")

    # pred_z = B_pred_z_adata.concatenate(CD14_Mono_pred_z_adata, CD4T_pred_z_adata, CD8T_pred_z_adata,
    #                                     Dendritic_pred_z_adata,
    #                                     FCGR3A_Mono_pred_z_adata, NK_pred_z_adata)
    pred_z = B_pred_z_adata.concatenate(CD14_Mono_pred_z_adata,
                                        Dendritic_pred_z_adata,
                                        NK_pred_z_adata)

    pred_control_z = control_z.concatenate(pred_z)

    sc.pp.neighbors(control_z)
    sc.tl.umap(control_z)
    sc.pl.umap(control_z, color=[opt['cell_type_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"_ctrl_z_{num_trial}_{opt['model_name']}.pdf",
               show=False,
               frameon=False)

    sc.pp.neighbors(control_z)
    sc.tl.umap(control_z)
    sc.pl.umap(control_z, color=[opt['cell_type_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"_ctrl_z_{num_trial}_{opt['model_name']}.pdf",
               show=False,
               frameon=False)


    sc.pp.neighbors(pred_z)
    sc.tl.umap(pred_z)
    sc.pl.umap(pred_z, color=[opt['cell_type_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"_pred_z_{num_trial}_{opt['model_name']}.pdf",
               show=False,
               frameon=False)

    sc.pp.neighbors(pred_control_z)
    sc.tl.umap(pred_control_z)
    sc.pl.umap(pred_control_z, color=[opt['cell_type_key'], opt['condition_key']],
               wspace=0.4,
               legend_fontsize=14,
               save=f"_all_z_{num_trial}_{opt['model_name']}.pdf",
               show=False,
               frameon=False)

    # knn acc
    print("===============knn acc========================")
    pred_condition = "pred_perturbed"
    if sparse.issparse(all_adata.X):
        X_train = all_adata[(all_adata.obs[opt['condition_key']] == opt['condition']['case'])].X.A
        y_train = (all_adata[(all_adata.obs[opt['condition_key']] == opt['condition']['case'])].obs[opt['cell_type_key']]).tolist()
        X_test = all_adata[(all_adata.obs[opt['condition_key']] == pred_condition)].X.A
        y_test = (all_adata[(all_adata.obs[opt['condition_key']] == pred_condition)].obs[opt['cell_type_key']]).tolist()
    else:
        X_train = all_adata[(all_adata.obs[opt['condition_key']] == opt['condition']['case'])].X
        y_train = (
        all_adata[(all_adata.obs[opt['condition_key']] == opt['condition']['case'])].obs[opt['cell_type_key']]).tolist()
        X_test = all_adata[(all_adata.obs[opt['condition_key']] == pred_condition)].X
        y_test = (all_adata[(all_adata.obs[opt['condition_key']] == pred_condition)].obs[opt['cell_type_key']]).tolist()
    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores = metrics.accuracy_score(y_test, y_pred)
    print("knn acc: " + str(scores))

    # DEGs
    print("===============DEGs========================")
    cell_type_list = adata.obs[opt['cell_type_key']].unique().tolist()
    total_DEGs = 0
    for cell_type in cell_type_list:
        ctrl_case_adata = adata
        ctrl_pred_adata = all_adata[(all_adata.obs[opt['cell_type_key']] == cell_type) &
                                    (all_adata.obs[opt['condition_key']].isin(
                                        [opt['condition']['control'], pred_condition]))]
        sc.tl.rank_genes_groups(ctrl_case_adata, groupby=opt['condition_key'], reference=opt['condition']['control'],
                                method='wilcoxon')
        sc.tl.rank_genes_groups(ctrl_pred_adata, groupby=opt['condition_key'], reference=opt['condition']['control'],
                                method='wilcoxon')

        ctrl_case_DEGs = ctrl_case_adata.uns['rank_genes_groups']['names'][opt['condition']['case']][0:100].tolist()
        ctrl_pred_DEGs = ctrl_pred_adata.uns['rank_genes_groups']['names'][pred_condition][0:100].tolist()
        same_DEGs = list(set(ctrl_case_DEGs).intersection(set(ctrl_pred_DEGs)))
        print("number of same DEGs for " + cell_type + ": " + str(len(same_DEGs)))
        total_DEGs += len(same_DEGs)
    print("number of total same DEGs: " + str(total_DEGs))


def train_whole(config, opt):
    adata = sc.read(opt['dataPath'])
    # cell_type_list = adata.obs[opt['cell_type_key']].unique().tolist()
    cell_type_list = ['CD14+ Monocyte', 'CD19+ B', 'Dendritic', 'CD56+ NK']

    print("cell type list: " + str(cell_type_list))
    for cell_type in cell_type_list:
        # print("=================" + cell_type + "=========================")
        opt['prediction_type'] = cell_type
        if not os.path.exists(opt['outf']):
            os.makedirs(opt['outf'])
        train_scPreGAN(config, opt)


def main(data_name, num_samples=10, gpus_per_trial=0):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if data_name == 'pbmc':
        # opt = {
        #     'cuda': True,
        #     'dataPath': '/home/wxj/scPreGAN-reproducibility/datasets/pbmc/pbmc-2000hvg.h5ad',
        #     'checkpoint_dir': None,
        #     'condition_key': 'condition',
        #     'condition': {"case": "stimulated", "control": "control"},
        #     'cell_type_key': 'cell_type',
        #     'prediction_type': None,
        #     'out_sample_prediction': True,
        #     'manual_seed': 3060,
        #     'data_name': 'pbmc',
        #     'model_name': 'pbmc_OOD_2000hvg_re',
        #     'outf': '/home/wxj/scPreGAN-reproducibility/datasets/pbmc/pbmc_OOD_AC_layer_S',
        #     'validation': False,
        #     'valid_dataPath': None,
        #     'use_sn': True,
        #     'use_wgan_div': True,
        #     'gan_loss': 'wgan',
        #     'train_flag': False,
        #     'n_classes': 7,
        # }
        opt = {
            'cuda': True,
            # 'dataPath': '../data/mlp_PBMC_seed42_1028-152104/train_pairedSplitCD19.h5ad', # CD14 / CD19
            # 'dataPath': 'C:\\Users\\kiria\\Desktop\\COMBINE_lab\\data\\mlp_PBMC_seed42_1028-152104\\train_pairedSplitCD19.h5ad',
            'dataPath': '.\\datasets\\train_pairedSplitCD19.h5ad',
            'checkpoint_dir': 'scpg_tune_chkp_cd19/', # cd14 / cd19
            'condition_key': 'KO_noKO',
            'condition': {"case": "KO", "control": "noKO"},
            'cell_type_key': 'celltype',
            'prediction_type': None,
            'out_sample_prediction': True,
            'manual_seed': 3060,
            'data_name': 'pbmc',
            'model_name': 'pbmc_OOD',
            'outf': 'scPreGANac_tunecd19/', # cd14 / cd19
            'validation': False,
            'valid_dataPath': None,
            'use_sn': True,
            'use_wgan_div': True,
            'gan_loss': 'wgan',
            'train_flag': True, # set to true!
            'n_classes': 8,
        }
    else:
        NotImplementedError()

    if cuda_is_available():
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        cudnn.benchmark = True

    config = {
        "lr_disc": tune.choice([0.001]),
        # "lr_e": tune.choice([0.001, 0.0001]),
        "lr_e": tune.choice([0.0001]),
        "lr_g": tune.choice([0.001]),
        "lambda_adv": tune.choice([0.1, 0.01, 0.001]),
        "lambda_recon": tune.choice([0.1, 1]),
        "lambda_encoding": tune.choice([1, 0.1]),
        "lambta_gp": tune.choice([0.1]),
        # "lambda_l1_reg": tune.choice([0]),
        "min_hidden_size": tune.choice([256]),
        "batch_size": tune.choice([64]),
        "z_dim": tune.choice([16, 32, 64, 128]),
        # "z_dim": tune.choice([16]),
        "niter": tune.choice([1000]
        )
    }

    if not os.path.exists(opt['outf']):
        os.makedirs(opt['outf'])
    reporter = CLIReporter(metric_columns=["D_A_loss", "D_B_loss", "adv_loss", "recon_loss", "encoding_loss", "G_loss",
                                           "adv_loss_val", "recon_loss_val", "encoding_loss_val", "G_loss",
                                           "D_A_loss_val", "D_B_loss_val", "training_iteration"])

    result = tune.run(
        partial(train_whole, opt=opt),
        name="hyper_scPreGAN",
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter)

    # train = sc.read(opt['dataPath'])
    # n_features = train.shape[1]

    for i in range(0, num_samples):
        print("============the ", i, "th trial:========================")
        trial_i = result.trials[i]
        test_results(opt, i, trial_i)


if __name__ == "__main__":

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main(data_name='pbmc', num_samples=10, gpus_per_trial=1)