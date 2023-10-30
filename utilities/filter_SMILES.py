import re, random
import numpy as np
from rdkit import Chem


def mol_to_smi_c(mol):
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def mol_to_elements(smi, mol):
    molecule = []
    for a in mol.GetAtoms():
        atom = a.GetSymbol()
        if atom not in molecule:
            molecule.append(atom)
    molecule = sorted(molecule, key=len, reverse=True)
    return list(filter(None, re.split(r'('+'|'.join(molecule)+'|\d|\(|\)|\#|\[|\]|\=|\.|\-|\+)', smi)))


smiles_length = 150
smiles_half_len = smiles_length / 2.0
data_prefix = '../data/'

elements = []
fh0 = open(data_prefix + 'Dictionary', 'r')
for line in fh0:
    elements.append(line.strip())
fh0.close()

fh0 = open(data_prefix + 'smiles.raw', 'r')
fh1 = open(data_prefix + 'smiles.train.csv', 'w')
fh2 = open(data_prefix + 'smiles.test.csv', 'w')

uniq_smi = []
i = 0
for line in fh0:
    smi = line.strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        smi_c = mol_to_smi_c(mol)
        try:
            molecule = mol_to_elements(smi_c, mol)
            if smi_c in uniq_smi or len(molecule) < np.random.normal(smiles_half_len + 15, 15, 1)[0]:
                prob = 0
            else:
                prob = 1
            if prob > 0 and len(molecule) <= smiles_length and all(element in elements for element in molecule):
                sample = smi_c
                for j in range(smiles_length):
                    if j < len(molecule):
                        sample += ',' + str(elements.index(molecule[j]))
                    else:
                        sample += ',0'
                sample += '\n'
                if random.random() < 0.8:
                    fh1.write(sample)
                else:
                    fh2.write(sample)
                i += 1
                uniq_smi.append(smi_c)
        except:
            pass
    if i == 1250000:
        break

fh0.close()
fh1.close()
fh2.close()
