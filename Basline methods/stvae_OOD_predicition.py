import scanpy as sc
import stvae
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy import sparse
from torch import zeros, Tensor, LongTensor, log, cuda, save, load
import torch
from torch.cuda import is_available as cuda_is_available
from scipy import sparse
import os
from pathlib import Path

data_name = 'pbmc'
data_path = '../data/mlp_PBMC_seed42_1028-152104/train_pairedSplitCD14.h5ad' # CD14 / CD19
if data_name == 'pbmc':
    adata = sc.read_h5ad(data_path)
    cell_type_key = 'celltype'
    condition_key = 'KO_noKO'
    condition = {"case": "KO", "control": "noKO"}
# elif data_name == 'hpoly':
#     adata = sc.read_h5ad("/home/wxj/scBranchGAN/datasets/Hpoly/hpoly.h5ad")
#     cell_type_key = 'cell_label'
#     condition_key = 'condition'
#     condition = {"case": "Hpoly.Day10", "control": "Control"}
# elif data_name == 'species':
#     adata = sc.read_h5ad("/home/wxj/scBranchGAN/datasets/species/species.h5ad")
#     cell_type_key = 'species'
#     condition_key = 'condition'
#     condition = {"case": "LPS6", "control": "unst"}
else:
    raise Exception("InValid data name")

cell_type_list = ['CD14+ Monocyte', 'CD19+ B', 'Dendritic', 'CD56+ NK']
print(cell_type_list)
for cell_type in cell_type_list:
    print("=================processing " + cell_type + "=================")
    train_set = adata[~((adata.obs[condition_key] == condition["case"]) & (adata.obs[cell_type_key] == cell_type))]
    if sparse.issparse(train_set.X):
        train_expr = train_set.X.A
    else:
        train_expr = train_set.X
    train_expr = train_expr if isinstance(train_expr, np.ndarray) else np.array(train_expr)
    train_condition = np.array(train_set.obs[condition_key].tolist())
    train_condition = OneHotEncoder(sparse=False).fit_transform(train_condition.reshape(-1, 1))
    train_labels = np.array(train_set.obs[cell_type_key].tolist())
    train_labels = OneHotEncoder(sparse=False).fit_transform(train_labels.reshape(-1, 1))
    
    # https://github.com/NRshka/stvae-source/blob/master/notebooks/using/training.ipynb
    cfg = stvae.Config()
    cfg.epochs = 100
    cfg.input_dim = train_expr.shape[1] # n_top_genes
    cfg.n_genes = train_expr.shape[1]
    cfg.count_labels = train_labels.shape[1]
    cfg.count_classes = train_condition.shape[1]
    
    cfg.noise_beta= 0.1
    cfg.decay_beta= 1.
    cfg.clip_value= 1.
    cfg.vae_lr = 0.001
    cfg.vae_decay = 1e-3
    cfg.disc_lr = 1e-4
    cfg.disc_decay = 0.
    cfg.vae_beta = 2e-5
    cfg.adv_weight = 0.0000001
    cfg.cyclic_weight = 0.2
    cfg.mmd_weight = 0
    cfg.l1_weight = 0#.5
    cfg.kernel_mu = 0.4
    cfg.model_scheduler_gamma = 0.992
    cfg.discr_scheduler_gamma = 0.992
    cfg.n_layers = 2
    cfg.scale_alpha = 1.3
    cfg.form_consistency_weight = 0.2

    cfg.bottleneck = 30
    # cfg.batch_size = 128
    # cfg.num_workers = 20
    cfg.activation = 'mish'
    cfg.condition_latent_dim = 10

    cfg.classifier_hidden_size = 512
    cfg.classifier_epochs = 200
    cfg.celltype_clf_lr = 1e-5
    cfg.form_clf_lr = 3e-4
    cfg.celltype_clf_wdecay = 0#weight decay
    cfg.form_clf_wdecay = 0#weight decay
    
    model = stvae.stVAE(cfg)
    indices = np.array(train_expr.shape[0] * [0])
    test_size = 2
    train_expression, val_expression, train_condition_ohe, val_condition_ohe, train_label_ohe, test_label_ohe, train_ind, test_ind = train_test_split(
        train_expr, train_condition, train_labels,
        indices,
        random_state=cfg.random_state,
        stratify=train_condition.argmax(1), test_size=test_size
    )
    model.train((train_expression, train_condition_ohe, train_label_ohe), None)
    ge_transfer_raw = adata[
        (adata.obs[condition_key] == condition['control']) & (adata.obs[cell_type_key] == cell_type)]
    if sparse.issparse(ge_transfer_raw.X):
        ge_transfer_raw = Tensor(ge_transfer_raw.X.A)
    else:
        ge_transfer_raw = Tensor(ge_transfer_raw.X)

    if (data_name == 'pbmc') or (data_name == 'hpoly'):
        source_classes = np.zeros(shape=(ge_transfer_raw.shape[0], 2))
        source_classes[:, 1] = 1
        target_classes = np.zeros(shape=(ge_transfer_raw.shape[0], 2))
        target_classes[:, 0] = 1
    elif (data_name == 'species') or (data_name == 'covid-19-pbmc'):
        source_classes = np.zeros(shape=(ge_transfer_raw.shape[0], 2))
        source_classes[:, 1] = 1
        target_classes = np.zeros(shape=(ge_transfer_raw.shape[0], 2))
        target_classes[:, 0] = 1
    else:
        raise Exception("InValid data name")

    source_classes = Tensor(source_classes)
    target_classes = Tensor(target_classes)

    if cfg.use_cuda and cuda_is_available():
        source_classes = source_classes.cuda()
        target_classes = target_classes.cuda()
        ge_transfer_raw = ge_transfer_raw.cuda()
    pred_expression_tensor = model.model(ge_transfer_raw, target_classes)[0]
    pred_expression_np = pred_expression_tensor[0].detach().cpu().numpy()
    pred_expression_np[pred_expression_np < 0] = 0
    pred_adata = sc.AnnData(X=pred_expression_np,
                            obs={condition_key: ["pred_perturbed"] * len(pred_expression_np),
                                 cell_type_key: [cell_type] * len(pred_expression_np)
                                 })
    pred_adata.var_names = adata.var_names
    
    save_path = f"./pred_data/stvae/{Path(data_path).stem}_pred"
    os.makedirs(save_path, exist_ok=True)
    pred_adata.write_h5ad(f"{save_path}/stvae_pred_{cell_type}.h5ad")
    
    del cfg
    del model
    del ge_transfer_raw
    del source_classes
    del target_classes
    torch.cuda.empty_cache()
print("training all finished")
