import torch
from torch.nn import functional as F


def loss_function(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')


def train(epoch, model, optimizer, device, log_interval, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, (_, data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_data = model(data)
        loss = loss_function(recon_data, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss on train set: {:.6f}'.format(epoch, train_loss))
    return train_loss


def test(epoch, model, device, test_loader):
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for i, (_, data) in enumerate(test_loader):
            data = data.to(device)
            recon_data = model(data)
            test_loss += loss_function(recon_data, data).item()
    test_loss /= len(test_loader.dataset)
    print('====> Epoch: {} Average loss on test set:  {:.6f}\n'.format(epoch, test_loss))
    return test_loss
