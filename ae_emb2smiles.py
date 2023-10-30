import torch
from ae.smiles import *


path_to_model = 'models/ae_model_best_test.pt'
model = torch.load(path_to_model)
model.eval()

path_to_dictionary = 'data/Dictionary'
smiles_dict = make_smiles_dict(path_to_dictionary)
number_of_columns = 150

for i in range(200000):
    emb_vector = torch.sigmoid(torch.rand(100))

    smiles, result = check_emb(emb_vector, model, smiles_dict, number_of_columns)
    if result > 0:
        print(str(i) + '\t' + smiles)
