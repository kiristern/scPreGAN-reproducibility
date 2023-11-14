from __future__ import print_function
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import mean, exp, unique, cat, isnan
from torch import norm as torch_norm

import multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.cuda import is_available as cuda_is_available
from torch import Tensor, FloatTensor
from torch.utils.data import random_split
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import scanpy as sc
import pandas as pd
import anndata
from scipy import sparse
from util import load_anndata, label_encoder
from model.Discriminator import Discriminator_AC
from model.Generator import Generator_AC_layer #, Generator_AC_layer_newCategorical
from model.Encoder import Encoder_AC_layer


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight, 1e-2)
        m.bias.data.fill_(0.01)


def create_model(n_features, z_dim, min_hidden_size, n_classes, use_cuda, use_sn, train_flag):
    D_A = Discriminator_AC(n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1, n_classes=n_classes)
    D_B = Discriminator_AC(n_features=n_features, min_hidden_size=min_hidden_size, out_dim=1, n_classes=n_classes)

    G_A = Generator_AC_layer(z_dim=z_dim, min_hidden_size=min_hidden_size, n_features=n_features)
    G_B = Generator_AC_layer(z_dim=z_dim, min_hidden_size=min_hidden_size, n_features=n_features)
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
    # load data===============================
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
    zeros = torch.zeros(config['batch_size'], 1)

    if (opt['cuda']) and torch.cuda.is_available():
        ones.cuda()
        zeros.cuda()

    D_A.train()
    D_B.train()
    G_A.train()
    G_B.train()
    E.train()

    D_A_loss = 0.0
    D_B_loss = 0.0

    log_path = f'./runs/{opt["data_name"]}/{opt["model_name"]}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    
    ############################################################# train
    if opt["train_flag"]:
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

            # Discriminator
            try:
                real_A, cell_type_A = next(A_train_loader_it)
                real_B, cell_type_B = next(B_train_loader_it)
            except StopIteration:
                A_train_loader_it, B_train_loader_it = iter(A_train_loader), iter(B_train_loader)
                real_A, cell_type_A = next(A_train_loader_it)
                real_B, cell_type_B = next(B_train_loader_it)

            if opt['cuda'] and cuda_is_available():
                real_A = real_A.cuda()
                real_B = real_B.cuda()
                cell_type_A = cell_type_A.cuda()
                cell_type_B = cell_type_B.cuda()

            D_A.zero_grad()
            D_B.zero_grad()

            out_A, out_A_cls = D_A(real_A, cell_type_A) # real A
            out_B, out_B_cls = D_B(real_B, cell_type_B) # real B

            real_A_z = E(real_A)
            AA = G_A(real_A_z)
            AB = G_B(real_A_z)
            
            real_B_z = E(real_B)
            BB = G_B(real_B_z)
            BA = G_A(real_B_z)

            out_BA, out_BA_cls = D_A(BA.detach(), cell_type_B) # false A
            out_AB, out_AB_cls = D_B(AB.detach(), cell_type_A) # false B

            _cell_type_A = torch.argmax(cell_type_A, dim=-1)
            _cell_type_B = torch.argmax(cell_type_B, dim=-1)
            
            dis_D_A_real = dis_criterion(out_A, torch.ones_like(out_A))
            aux_D_A_real = aux_criterion(out_A_cls, _cell_type_A)
            D_A_real = dis_D_A_real + aux_D_A_real
            dis_D_A_fake = dis_criterion(out_BA, torch.zeros_like(out_BA))
            aux_D_A_fake = aux_criterion(out_BA_cls, _cell_type_B) 
            D_A_fake = dis_D_A_fake + aux_D_A_fake

            dis_D_B_real = dis_criterion(out_B, torch.ones_like(out_B))
            aux_D_B_real = aux_criterion(out_B_cls, _cell_type_B)
            D_B_real = dis_D_B_real + aux_D_B_real
            dis_D_B_fake = dis_criterion(out_AB, torch.zeros_like(out_AB)) 
            aux_D_B_fake = aux_criterion(out_AB_cls, _cell_type_A) 
            D_B_fake = dis_D_B_fake + aux_D_B_fake

            D_A_loss = D_A_real + D_A_fake
            D_B_loss = D_B_real + D_B_fake
            # print("D_A_loss: " + str(D_A_loss.item()))

            # Calculate discriminator accuracy
            pred = np.concatenate([out_BA_cls.data.cpu().numpy(),  out_A_cls.data.cpu().numpy()], axis=0)
            gt = np.concatenate([_cell_type_B.cpu().numpy(),  _cell_type_A.data.cpu().numpy()], axis=0)
            acc_A = np.mean(np.argmax(pred, axis=1) == gt)

            pred = np.concatenate([out_AB_cls.data.cpu().numpy(), out_B_cls.data.cpu().numpy()], axis=0)
            gt = np.concatenate([_cell_type_A.cpu().numpy(), _cell_type_B.data.cpu().numpy()], axis=0)
            acc_B = np.mean(np.argmax(pred, axis=1) == gt)


            pred = np.concatenate([out_BA.data.cpu().numpy(), out_A.data.cpu().numpy()], axis=0)
            gt = np.concatenate([zeros.cpu().numpy(),  ones.data.cpu().numpy()], axis=0)
            acc_A_D = np.mean((pred>0.5) == gt)

            pred = np.concatenate([out_AB.data.cpu().numpy(),  out_B.data.cpu().numpy()], axis=0)
            gt = np.concatenate([zeros.cpu().numpy(), ones.data.cpu().numpy()], axis=0)
            acc_B_D = np.mean((pred>0.5) == gt)

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

            G_AA_adv_loss = dis_criterion(out_AA, torch.ones_like(out_AA)) + aux_criterion(out_AA_cls, _cell_type_A)
            G_BA_adv_loss = dis_criterion(out_BA, torch.ones_like(out_BA)) + aux_criterion(out_BA_cls, _cell_type_B)
            G_ABA_adv_loss = dis_criterion(out_ABA, torch.ones_like(out_ABA)) + aux_criterion(out_ABA_cls, _cell_type_A)
        
            G_BB_adv_loss = dis_criterion(out_BB, torch.ones_like(out_BB)) + aux_criterion(out_BB_cls, _cell_type_B)
            G_AB_adv_loss = dis_criterion(out_AB, torch.ones_like(out_AB)) + aux_criterion(out_AB_cls, _cell_type_A)
            G_BAB_adv_loss = dis_criterion(out_BAB, torch.ones_like(out_BAB)) + aux_criterion(out_BAB_cls, _cell_type_B)

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

            writer.add_scalar('D_A_loss', D_A_loss, global_step=iteration)
            writer.add_scalar('D_B_loss', D_B_loss, global_step=iteration)
            writer.add_scalar('adv_loss', adv_loss, global_step=iteration)
            writer.add_scalar('recon_loss', recon_loss, global_step=iteration)
            writer.add_scalar('encoding_loss', encoding_loss, global_step=iteration)
            writer.add_scalar('G_loss', G_loss, global_step=iteration)

            if iteration % 100 == 0:
                print(
                    '[%d/%d] D_A_loss: %.4f  D_B_loss: %.4f adv_loss: %.4f  recon_loss: %.4f encoding_loss: %.4f G_loss: %.4f'
                    % (iteration, config['niter'], D_A_loss.item(), D_B_loss.item(), adv_loss.item(), recon_loss.item(),
                    encoding_loss.item(), G_loss.item()))

                if opt['validation']:
                    D_A_loss_val = 0.0
                    D_B_loss_val = 0.0

                    adv_loss_val = 0.0
                    recon_loss_val = 0.0
                    encoding_loss_val = 0.0
                    G_loss_val = 0.0

                    counter = 0
                    sparcity_cellA_val = 0.0
                    sparcity_cellB_val = 0.0

                    A_valid_loader_it = iter(A_valid_loader)
                    B_valid_loader_it = iter(B_valid_loader)

                    max_length = max(len(A_valid_loader), len(B_valid_loader))
                    with torch.no_grad():
                        for iteration_val in range(1, max_length):
                            try:
                                cellA_val, cell_type_A_val = next(A_valid_loader_it)[0]
                                cellB_val, cell_type_B_val  = next(B_valid_loader_it)[0]
                            except StopIteration:
                                A_valid_loader_it, B_valid_loader_it = iter(A_valid_loader), iter(B_valid_loader)
                                cellA_val, cell_type_A_val = next(A_valid_loader_it)[0]
                                cellB_val, cell_type_B_val = next(B_valid_loader_it)[0]

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

                            outA_val, out_A_cls = D_A(cellA_val, cell_type_A_val)
                            outB_val, out_B_cls = D_B(cellB_val, cell_type_B_val)
                            out_AA, out_AA_cls = D_A(AA, cell_type_A_val)
                            out_BB, out_BB_cls = D_B(BB, cell_type_B_val)
                            out_AB, out_AB_cls = D_B(AB, cell_type_A_val)
                            out_BA, out_BA_cls = D_A(BA, cell_type_B_val)
                            out_ABA, out_ABA_cls = D_A(ABA, cell_type_A_val)
                            out_BAB, out_BAB_cls = D_B(BAB, cell_type_B_val)

                            _cell_type_A = torch.argmax(cell_type_A_val, dim=-1)
                            _cell_type_B = torch.argmax(cell_type_B_val, dim=-1)

                            D_A_real_loss_val = dis_criterion(outA_val, ones) + aux_criterion(out_A_cls, _cell_type_A)
                            D_B_real_loss_val = dis_criterion(outB_val, ones) + aux_criterion(out_B_cls, _cell_type_B)
                            D_A_fake_loss_val = dis_criterion(out_BA, zeros) + aux_criterion(out_BA_cls, _cell_type_B) 
                            D_B_fake_loss_val = dis_criterion(out_AB, zeros) + aux_criterion(out_AB_cls, _cell_type_A) 

                            D_A_loss_val += (D_A_real_loss_val + D_A_fake_loss_val).item()
                            D_B_loss_val += (D_B_real_loss_val + D_B_fake_loss_val).item()

                            G_AA_adv_loss_val = dis_criterion(out_AA, ones) + aux_criterion(out_AA_cls, _cell_type_A)
                            G_BA_adv_loss_val  = dis_criterion(out_BA, ones) + aux_criterion(out_BA_cls, _cell_type_B)
                            G_ABA_adv_loss_val  = dis_criterion(out_ABA, ones) + aux_criterion(out_ABA_cls, _cell_type_A)
                        
                            G_BB_adv_loss_val  = dis_criterion(out_BB, ones) + aux_criterion(out_BB_cls, _cell_type_B)
                            G_AB_adv_loss_val  = dis_criterion(out_AB, ones) + aux_criterion(out_AB_cls, _cell_type_A)
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
                            encoding_loss_val = (
                                                        l_encoding_AA_val + l_encoding_BB_val + l_encoding_BA_val + l_encoding_AB_val).item() * \
                                                config['lambda_encoding']
                            G_loss_val += adv_loss_val + recon_loss_val + encoding_loss_val

                            cellA_val_np = cellA_val.cpu().detach().numpy()
                            cellB_val_np = cellB_val.cpu().detach().numpy()

                            cellA_val_zero = cellA_val_np[cellA_val_np <= 0]
                            cellB_val_zero = cellB_val_np[cellB_val_np <= 0]

                            sparcity_cellA_val += cellA_val_zero.shape[0] / (cellA_val_np.shape[0] * cellA_val_np.shape[1])
                            sparcity_cellB_val += cellB_val_zero.shape[0] / (cellB_val_np.shape[0] * cellB_val_np.shape[1])

                    print(
                        '[%d/%d] adv_loss_val: %.4f  recon_loss_val: %.4f encoding_loss_val: %.4f  G_loss: %.4f D_A_loss_val: %.4f D_B_loss_val: %.4f'
                        % (iteration, config['niter'], adv_loss_val / counter, recon_loss_val / counter,
                        encoding_loss_val / counter, G_loss / counter, D_A_loss_val / counter, D_B_loss_val / counter))

                    writer.add_scalar('adv_loss_val', adv_loss_val / counter, global_step=iteration)
                    writer.add_scalar('recon_loss_val', recon_loss_val / counter, global_step=iteration)
                    writer.add_scalar('encoding_loss_val', encoding_loss_val / counter, global_step=iteration)
                    writer.add_scalar('G_loss', G_loss / counter, global_step=iteration)
                    writer.add_scalar('D_A_loss_val', D_A_loss_val / counter, global_step=iteration)
                    writer.add_scalar('D_B_loss_val', D_B_loss_val / counter, global_step=iteration)

                    print(
                        '[%d/%d] adv_loss_val: %.4f  recon_loss_val: %.4f encoding_loss_val: %.4f  G_loss: %.4f D_A_loss_val: %.4f D_B_loss_val: %.4f'
                        % (iteration, config['niter'], adv_loss_val / counter, recon_loss_val / counter,
                        encoding_loss_val / counter, G_loss / counter, D_A_loss_val / counter, D_B_loss_val / counter))


            if opt['checkpoint_dir'] is not None and iteration % 10000 == 0:
                path = os.path.join(opt['checkpoint_dir'], "checkpoint_E.pth")
                torch.save((E.state_dict(), optimizerE.state_dict()), path)
                path = os.path.join(opt['checkpoint_dir'], "checkpoint_G_A.pth")
                torch.save((G_A.state_dict(), optimizerG_A.state_dict()), path)
                path = os.path.join(opt['checkpoint_dir'], "checkpoint_G_B.pth")
                torch.save((G_B.state_dict(), optimizerG_B.state_dict()), path)
                path = os.path.join(opt['checkpoint_dir'], "checkpoint_D_A.pth")
                torch.save((D_A.state_dict(), optimizerD_A.state_dict()), path)
                path = os.path.join(opt['checkpoint_dir'], "checkpoint_D_B.pth")
                torch.save((D_B.state_dict(), optimizerD_B.state_dict()), path)

        torch.save(E, os.path.join(opt['outf'], f'E_{opt["prediction_type"]}.pth'))
        torch.save(G_A, os.path.join(opt['outf'], f'G_A_{opt["prediction_type"]}.pth'))
        torch.save(G_B, os.path.join(opt['outf'], f'G_B_{opt["prediction_type"]}.pth'))
        torch.save(D_A, os.path.join(opt['outf'], f'D_A_{opt["prediction_type"]}.pth'))
        torch.save(D_B, os.path.join(opt['outf'], f'D_B_{opt["prediction_type"]}.pth'))
        writer.close()
    else:
        E = torch.load(os.path.join(opt['outf'], f'E_{opt["prediction_type"]}.pth'))
        G_A = torch.load(os.path.join(opt['outf'], f'G_A_{opt["prediction_type"]}.pth'))
        G_B = torch.load(os.path.join(opt['outf'], f'G_B_{opt["prediction_type"]}.pth'))
        D_A = torch.load(os.path.join(opt['outf'], f'D_A_{opt["prediction_type"]}.pth'))
        D_B = torch.load(os.path.join(opt['outf'], f'D_B_{opt["prediction_type"]}.pth'))
    print("Finished Training")



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

    if opt['cuda'] and cuda_is_available():
        control_tensor = control_tensor.cuda()
        cell_type_control_tensor = cell_type_control_tensor.cuda()
        case_tensor = case_tensor.cuda()
        cell_type_case_tensor = cell_type_case_tensor.cuda()
    control_z = E(control_tensor)
    case_z = E(case_tensor)
    # case_pred = G_B(control_z, cell_type_control_tensor) # https://github.com/JaneJiayiDong/scPreGAN/blob/3d6f3a6631963a30c82a48621eba6b009cce7d17/scPreGAN/model/scPreGAN.py#L459C20-L459C20
    case_pred = G_B(control_z)
    real_case_pred = G_B(case_z)

    # control的隐向量
    control_z_adata = anndata.AnnData(X=control_z.cpu().detach().numpy(),
                                      obs={opt['condition_key']: ["control_z"] * len(control_z),
                                           opt['cell_type_key']: control_adata.obs[opt['cell_type_key']].tolist()})
    if not os.path.exists(os.path.join(opt['outf'], 'control_z_adata')):
        os.makedirs(os.path.join(opt['outf'], 'control_z_adata'))
    control_z_adata.write_h5ad(os.path.join(opt['outf'], 'control_z_adata', f'control_z_{opt["prediction_type"]}.h5ad'))

    # case的隐向量
    case_z_adata = anndata.AnnData(X=case_z.cpu().detach().numpy(),
                                      obs={opt['condition_key']: ["case_z"] * len(case_z),
                                           opt['cell_type_key']: case_adata.obs[opt['cell_type_key']].tolist()})
    if not os.path.exists(os.path.join(opt['outf'], 'case_z_adata')):
        os.makedirs(os.path.join(opt['outf'], 'case_z_adata'))
    case_z_adata.write_h5ad(os.path.join(opt['outf'], 'case_z_adata', f'case_z_{opt["prediction_type"]}.h5ad'))

    # 真实case预测
    real_perturbed_adata = anndata.AnnData(X=real_case_pred.cpu().detach().numpy(),
                                           obs={opt['condition_key']: ["pred_perturbed"] * len(real_case_pred),
                                                opt['cell_type_key']: case_adata.obs[opt['cell_type_key']].tolist()})
    real_perturbed_adata.var_names = adata.var_names
    if not os.path.exists(os.path.join(opt['outf'], 'real_case_adata')):
        os.makedirs(os.path.join(opt['outf'], 'real_case_adata'))
    real_perturbed_adata.write_h5ad(os.path.join(opt['outf'], 'real_case_adata', f'real_case_{opt["prediction_type"]}.h5ad'))

    # 预测数据
    pred_perturbed_adata = anndata.AnnData(X=case_pred.cpu().detach().numpy(),
                                           obs={opt['condition_key']: ["pred_perturbed"] * len(case_pred),
                                                opt['cell_type_key']: control_adata.obs[opt['cell_type_key']].tolist()})
    pred_perturbed_adata.var_names = adata.var_names
    if not os.path.exists(os.path.join(opt['outf'], 'pred_adata/')):
        os.makedirs(os.path.join(opt['outf'], 'pred_adata/'))
    pred_perturbed_adata.write_h5ad(os.path.join(opt['outf'], 'pred_adata/', f'scpGAN_pred_{opt["prediction_type"]}.h5ad'))

    # 预测数据的隐向量
    pred_z = E(case_pred)
    pred_z_adata = anndata.AnnData(X=pred_z.cpu().detach().numpy(),
                                   obs={opt['condition_key']: ["pred_z"] * len(pred_z),
                                        opt['cell_type_key']: pred_perturbed_adata.obs[opt['cell_type_key']].tolist()})
    if not os.path.exists(os.path.join(opt['outf'], '/pred_z_adata')):
        os.makedirs(os.path.join(opt['outf'], '/pred_z_adata'))
    pred_z_adata.write_h5ad(
        os.path.join(opt['outf'], '/pred_z_adata', f'scpGAN_pred_z_{opt["prediction_type"]}.h5ad'))


def main(data_name):
    if data_name == 'pbmc':
        opt = {
            'cuda': True,
            'dataPath': '../data/mlp_PBMC_seed42_1028-152104/train_pairedSplitCD19.h5ad', # CD14 / CD19
            'checkpoint_dir': None, #'scPreGANac_chkp_cd19/', # cd14 / cd19
            'condition_key': 'KO_noKO',
            'condition': {"case": "KO", "control": "noKO"},
            'cell_type_key': 'celltype',
            'prediction_type': None,
            'out_sample_prediction': True,
            'manual_seed': 3060,
            'data_name': 'pbmc',
            'model_name': 'pbmc_OOD',
            'outf': 'scPreGANac_predcd19_temp/', # cd14 / cd19
            'validation': False,
            'valid_dataPath': None,
            'use_sn': True,
            'use_wgan_div': True,
            'gan_loss': 'wgan',
            'train_flag': True, # set to true!
            'n_classes': 8,
        }
    elif data_name == 'hpoly':
        opt = {
            'cuda': True,
            'dataPath': '/home/wxj/scPreGAN/datasets/Hpoly/hpoly-2000hvg.h5ad',
            'checkpoint_dir': None,
            'condition_key': 'condition',
            'condition': {"case": "Hpoly.Day10", "control": "Control"},
            'cell_type_key': 'cell_label',
            'prediction_type': None,
            'out_sample_prediction': True,
            'manual_seed': 3060,
            'data_name': 'hpoly',
            'model_name': 'hpoly_OOD',
            'outf': '/home/wxj/scPreGAN/datasets/Hpoly/hpoly_OOD',
            'validation': False,
            'valid_dataPath': None,
            'use_sn': True,
            'use_wgan_div': True,
            'gan_loss': 'wgan',
            'train_flag': True,
            'n_classes': 8,
        }
    elif data_name == 'species':
        opt = {
            'cuda': True,
            'dataPath': '/home/wxj/scPreGAN/datasets/species/species-2000hvg.h5ad',
            'checkpoint_dir': None,
            'condition_key': 'condition',
            'condition': {"case": "LPS6", "control": "unst"},
            'cell_type_key': 'species',
            'prediction_type': None,
            'out_sample_prediction': True,
            'manual_seed': 3060,
            'data_name': 'species',
            'model_name': 'species_OOD',
            'outf': '/home/wxj/scPreGAN/datasets/species/species_OOD',
            'validation': False,
            'valid_dataPath': None,
            'use_sn': True,
            'use_wgan_div': True,
            'gan_loss': 'wgan',
            'train_flag': True,
            'n_classes': 4,
        }
    elif data_name == 'covid':
        opt = {
            'cuda': True,
            'dataPath': '/home/wxj/scPreGAN-reproducibility/datasets/covid-19/covid-19-filtered-hvg2000.h5ad',
            'checkpoint_dir': None,
            'condition_key': 'Status',
            'condition': {"case": "COVID", "control": "Healthy"},
            'cell_type_key': 'cell_type_coarse',
            'prediction_type': None,
            'out_sample_prediction': True,
            'manual_seed': 3060,
            'data_name': 'covid',
            'model_name': 'covid_OOD_200hvg',
            'outf': '/home/wxj/scPreGAN-reproducibility/datasets/covid/covid_OOD_2000hvg',
            'validation': False,
            'valid_dataPath': None,
            'use_sn': True,
            'use_wgan_div': True,
            'gan_loss': 'wgan',
            'train_flag': True,
            'n_classes': 10,
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
        "batch_size": 64,
        "lambda_adv": 0.001,
        "lambda_encoding": 0.1,
        "lambda_l1_reg": 0,
        "lambda_recon": 1,
        "lambta_gp": 1,
        "lr_disc": 0.001,
        "lr_e": 0.0001,
        "lr_g": 0.001,
        "min_hidden_size": 256,
        "niter": 20000,
        "z_dim": 16
    }

    opt['out_sample_prediction'] = True
    adata = sc.read(opt['dataPath'])
    # cell_type_list = adata.obs[opt['cell_type_key']].unique().tolist()
    cell_type_list = ['CD14+ Monocyte', 'CD19+ B', 'Dendritic', 'CD56+ NK']
    print("cell type list: " + str(cell_type_list))
    for cell_type in cell_type_list:
        print("=================" + cell_type + "=========================")
        opt['prediction_type'] = cell_type

        if not os.path.exists(opt['outf']):
            os.makedirs(opt['outf'])
        train_scPreGAN(config, opt)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main(data_name='pbmc')