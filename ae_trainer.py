import argparse
import os
import shutil
import torch
from ae.smiles import *
from ae.autoencoder import *
from ae.modeling import *


parser = argparse.ArgumentParser(description='SMILES AutoEncoder')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=0, metavar='N',
                    help='number of epochs to train (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before printing training status')
parser.add_argument('--save-model-every', type=int, default=2, metavar='N',
                    help='how many epochs to wait before saving trained model (default: 2)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='F',
                    help='learning rate (default: 1e-3)')

args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device('cpu')

kwargs = {}

log_prefix = 'logs/'
model_prefix = 'models/'
data_prefix = 'data/'

for dir in [log_prefix, model_prefix]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

path_to_current_model = model_prefix + 'ae_model.pt'
path_to_current_model_log = log_prefix + 'ae_model.log'

path_to_best_train_model = model_prefix + 'ae_model_best_train.pt'
path_to_best_train_model_log = log_prefix + 'ae_model_best_train.log'

path_to_best_test_model = model_prefix + 'ae_model_best_test.pt'
path_to_best_test_model_log = log_prefix + 'ae_model_best_test.log'

path_to_dictionary = data_prefix + 'Dictionary'

smiles_dict = make_smiles_dict(path_to_dictionary)
size_of_smiles_dict = len(smiles_dict)

train_dataset = SMILESDataset(path_to_csv_file=data_prefix + 'smiles.train.csv', size_of_smiles_dict=size_of_smiles_dict)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
print(str(len(train_loader.dataset)) + ' samples loaded as the train dataset')

test_dataset = SMILESDataset(path_to_csv_file=data_prefix + 'smiles.test.csv', size_of_smiles_dict=size_of_smiles_dict)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
print(str(len(test_loader.dataset)) + ' samples loaded as the test dataset\n')

number_of_columns = train_dataset.col_num()

if os.path.isfile(path_to_current_model):
    model = torch.load(path_to_current_model)
else:
    model = AE().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch, model, optimizer, device, args.log_interval, train_loader)
    test_loss = test(epoch, model, device, test_loader)

    model_saved = False
    fh = open(path_to_current_model_log, 'a')
    fh.write(str(train_loss) + '\t' + str(test_loss) + '\t' + str(scheduler.get_last_lr()[0]) + '\n')
    fh.close()
    if epoch % args.save_model_every == 0 or epoch == args.epochs:
        torch.save(model, path_to_current_model)
        model_saved_path = path_to_current_model
        model_saved = True
        print('model saved to [ ' + path_to_current_model + ' ]\n')

    # assessing best lossess
    if epoch == 1:
        idle_counter = 0
        if os.path.isfile(path_to_best_train_model_log):
            fh = open(path_to_best_train_model_log, 'r')
            train_loss_best = float(fh.readline())
            fh.close()
        else:
            train_loss_best = train_loss

        if os.path.isfile(path_to_best_test_model_log):
            fh = open(path_to_best_test_model_log, 'r')
            _ = fh.readline()
            test_loss_best = float(fh.readline())
            fh.close()
        else:
            test_loss_best = test_loss

    # saving model with best train loss
    if train_loss < train_loss_best:
        if model_saved:
            shutil.copyfile(model_saved_path, path_to_best_train_model)
        else:
            torch.save(model, path_to_best_train_model)
            model_saved_path = path_to_best_train_model
            model_saved = True
        fh = open(path_to_best_train_model_log, 'w')
        fh.write(str(train_loss) + '\n' + str(test_loss) + '\n')
        fh.close()
        train_loss_best = train_loss
        print('model with best train loss saved to [ \x1b[1;34m' + path_to_best_train_model + '\x1b[0m ]\n')

    # saving model with best test loss
    if test_loss < test_loss_best:
        if model_saved:
            shutil.copyfile(model_saved_path, path_to_best_test_model)
        else:
            torch.save(model, path_to_best_test_model)
            model_saved_path = path_to_best_test_model
            model_saved = True
        fh = open(path_to_best_test_model_log, 'w')
        fh.write(str(train_loss) + '\n' + str(test_loss) + '\n')
        fh.close()
        test_loss_best = test_loss
        print('model with best test loss saved to [ \x1b[1;32m' + path_to_best_test_model + '\x1b[0m ]\n')
        idle_counter = 0
    else:
        idle_counter += 1
        if idle_counter % 3 == 0:
            scheduler.step()
            print('new learning rate: ' + str(scheduler.get_last_lr()[0]))

    if idle_counter > 6:
        break
