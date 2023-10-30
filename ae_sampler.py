import os
import torch
from rdkit import Chem, rdBase
from ae.smiles import *

kwargs = {}

path_to_model = 'models/ae_model_best_test.pt'
path_to_dictionary = 'data/Dictionary'

if os.path.isfile(path_to_model):
    model = torch.load(path_to_model)

    smiles_dict = make_smiles_dict(path_to_dictionary)
    size_of_smiles_dict = len(smiles_dict)

    sample_dataset = SMILESDataset(path_to_csv_file='data/smiles.test.csv', size_of_smiles_dict=size_of_smiles_dict)
    number_of_columns = sample_dataset.col_num()
    sample_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=1, shuffle=False, **kwargs)

    model.eval()
    n_bad = 0
    n_good = 0
    n_exact = 0
    rdBase.DisableLog('rdApp.error')
    for (label, data) in sample_loader:
        with torch.no_grad():
            recon_data = model(data)
        for (lbl, item) in zip(label, recon_data):
            out_lbl = onehot_to_smiles(item, smiles_dict, number_of_columns)
            if Chem.MolFromSmiles(out_lbl) is not None:
                n_good += 1
                if lbl == out_lbl:
                    n_exact += 1
            else:
                n_bad += 1
    print('bad   - ' + str(n_bad))
    print('good  - ' + str(n_good))
    print('exact - ' + str(n_exact))
else:
    print('Error: Model not found, check the path ', path_to_model)
