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
    rdBase.DisableLog('rdApp.error')
    fh_emb = open('test_embeddings.csv', 'w')
    for (label, data) in sample_loader:
        with torch.no_grad():
            emb_data = model.encode(data)
            recon_data = model.decode(emb_data)
        for (lbl, item, emb) in zip(label, recon_data, emb_data):
            out_lbl = onehot_to_smiles(item, smiles_dict, number_of_columns)
            if Chem.MolFromSmiles(out_lbl) is not None:
                fh_emb.write(lbl)
                for e in emb.tolist():
                    fh_emb.write(',' + str(e))
                fh_emb.write('\n')
    fh_emb.close()

else:
    print('Error: Model not found, check the path ', path_to_model)
