import sys
import logging
import scanpy as sc
import scgen
import os
from pathlib import Path

logger = logging.getLogger("scvi.inference.autotune")
logger.setLevel(logging.WARNING)

data_path = '../data/mlp_PBMC_seed42_1028-152104/train_pairedSplitCD14.h5ad' # CD19 / CD14 
data_name = "pbmc"
if data_name == "pbmc":
    # train = sc.read("datasets/train_pbmc.h5ad")
    train = sc.read(data_path)
    cell_type_key = "celltype"
    condition_key = "KO_noKO"
    condition = {"case": "KO", "control": "noKO"}
elif data_name == 'hpoly':
    train = sc.read("/home/wxj/scBranchGAN/datasets/Hpoly/hpoly.h5ad")
    cell_type_key = 'cell_label'
    condition_key = 'condition'
    condition = {"case": "Hpoly.Day10", "control": "Control"}
elif data_name == 'species':
    train = sc.read("/home/wxj/scBranchGAN/datasets/species/species.h5ad")
    cell_type_key = 'species'
    condition_key = 'condition'
    condition = {"case": "LPS6", "control": "unst"}
else:
    raise Exception("InValid data name")

cell_type_list = train.obs[cell_type_key].unique().tolist()
for cell_type in ['CD14+ Monocyte', 'CD19+ B', 'Dendritic', 'CD56+ NK']:
    print("=================processing " + cell_type + "=================")
    train_new = train[~((train.obs[cell_type_key] == cell_type) &
                        (train.obs[condition_key] == condition["case"]))]
    train_new = scgen.setup_anndata(train_new, copy=True, batch_key=condition_key, labels_key=cell_type_key)
    model = scgen.SCGEN(train_new)
    model.train(max_epochs=100, batch_size=32, early_stopping=True, early_stopping_patience=25)
    pred_adata, delta = model.predict(ctrl_key=condition["control"], stim_key=condition["case"],
                                      celltype_to_predict=cell_type)
    pred_adata.obs['condition'] = 'pred_perturbed'
    
    save_path = f"./pred_data/scgen/{Path(data_path).stem}_pred"
    os.makedirs(save_path, exist_ok=True)
    pred_adata.write_h5ad(f"{save_path}/scgen_pred_{cell_type}.h5ad")
