import torch
import torch.utils.data
import re
import math
from rdkit import Chem
from rdkit import rdBase


class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_csv_file, size_of_smiles_dict):
        fh = open(path_to_csv_file, 'r')

        self.labels = []
        self.onehots = []

        for line in fh:
            label, onehot_r, onehot_l, onehot_c = csv_to_onehot(line, size_of_smiles_dict)
            self.labels.append(label)
            self.onehots.append(onehot_r)
            self.labels.append(label)
            self.onehots.append(onehot_l)
            self.labels.append(label)
            self.onehots.append(onehot_c)
        fh.close()

        self.length = len(self.labels)
        self.number_of_columns = self.onehots[0].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.labels[index], self.onehots[index]

    def col_num(self):
        return self.number_of_columns


def make_smiles_dict(path_to_dictionary):
    smiles_dict = {}
    fh = open(path_to_dictionary, 'r')
    for i, line in enumerate(fh):
        smiles_dict[i] = line.strip()
    fh.close()
    return smiles_dict


def onehot_to_smiles(onehot, smiles_dict, number_of_columns):
    smiles = ''
    for i in range(number_of_columns):
        smiles += smiles_dict[int(onehot[i].max(0)[1])]
    return smiles.strip('0')


def csv_to_onehot(csv_line, size_of_smiles_dict):
    arr_r = csv_line.strip().split(',')
    label = arr_r[0]
    arr_r = arr_r[1:]

    arr_nonzero = arr_r.copy()
    while '0' in arr_nonzero: arr_nonzero.remove('0')

    num_zeros = len(arr_r) - len(arr_nonzero)
    arr_l = ['0'] * num_zeros + arr_nonzero
    arr_c = ['0'] * math.floor(0.5 * num_zeros) + arr_nonzero + ['0'] * math.ceil(0.5 * num_zeros)

    onehot_r = torch.zeros(len(arr_r), size_of_smiles_dict)
    onehot_l = torch.zeros(len(arr_l), size_of_smiles_dict)
    onehot_c = torch.zeros(len(arr_c), size_of_smiles_dict)
    for i in range(len(arr_r)):
        onehot_r[i, int(arr_r[i])] = 1
        onehot_l[i, int(arr_l[i])] = 1
        onehot_c[i, int(arr_c[i])] = 1
    return label, onehot_r, onehot_l, onehot_c


def check_with_rdkit(smiles_string):
    rdBase.DisableLog('rdApp.error')
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is not None:
        return 1
    else:
        return 0


def check_emb(embedding_vector, model, smiles_dict, number_of_columns):
    with torch.no_grad():
        onehot = model.decode(embedding_vector.view(1, 100))[0]
    smiles = onehot_to_smiles(onehot, smiles_dict, number_of_columns)
    return smiles, check_with_rdkit(smiles)

