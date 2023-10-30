# AutoEncoder for SMILES with embedding size of 100

This repo includes a set of Python scripts for molecular autoencoder. The PyTorch framework is used in this implementation. We consider the class of small organic molecules and use SMILES notation. The autoencoder's task is to encode SMILES string to some embedding compressed vector and then decode it back to the original SMILES.

The SMILES strings are canonicalized, kekulized, and one-hot encoded according to a dictionary of symbols. The autoencoder has a limitation of 150 symbol for the SMILES length, thus, shorter SMILES strings are zero-padded (in three ways -- left, right, both ends).

The dictionaly of unique symbols consists of 21 elements (`data/Dictionary`):
‘0’, ‘1’, ‘2’, ‘3’, ‘4’, ‘5’, ‘6’, ‘=’, ‘#’, ‘(’, ‘)’, ‘C’, ‘N’, ‘O’, ‘P’, ‘S’, ‘Na’, ‘F’, ‘Cl’, ‘Br’, ‘I’

`utilities/filter_SMILES.py` script filters SMILES, splits the data to train/test subsets, and performs one-hot encoding over them.

`ae_trainer.py` trains the autoencoder. During training it writes logs to `logs/` and saves current/best models to the `models/`.

Once models are trained, `ae_sampler.py` can be used to compare input molecules vs output ones.

`ae_smiles2emb.py` and `ae_emb2smiles.py` are used to convert SMILES strings to embeddings and vice versa.
