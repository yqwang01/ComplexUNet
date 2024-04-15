from cmath import inf
import shutil
import os
import torch
import torch.nn.modules.loss as Loss
from torch import optim
import numpy as np
from torch.autograd import Variable
from net.cmplx_unet import CUNet
from net.real_unet import RealUNet
from net.cmplx_blocks import batch_norm
from utils.dataset import get_dataloaders
from utils.loss import SSIM
from configs import config
import json
import random
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity as ssim



def set_seeds(seed):
    """Set the seeds for reproducibility
    
    Parameters
    ----------
    seed : int
        The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get device

    Returns
    -------
    device : torch.device
        The device to use.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(net, optimizer, loss_criterion, tr_dataloader, epoch, device):
    """Train for one epoch of the data

    Parameters
    ----------
    net : torch.nn.Module
        The network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    loss_criterion : torch.nn.Module
        The loss criterion to use.
    tr_dataloader : torch.utils.data.DataLoader
        The training data loader.
    epoch : int
        The epoch number.

    Returns
    -------
    avg_loss : float
        The average loss for the epoch.
    net : torch.nn.Module
        The trained network.
    optimizer : torch.optim.Optimizer
        The optimizer.
    """
    avg_loss = 0.0
    net.train()
    # radial_normalizer = batch_norm(
    #     in_channels = 1,
    # )
    for itt, (input, target, name) in enumerate(tr_dataloader):
        X = input.to(device)
        y = target.to(device)

        # if config.normalize_input:
        #     X = radial_normalizer(X)
        #     y = radial_normalizer(y)

        y_pred = net(X)

        loss = loss_criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().item() / len(tr_dataloader)

        if itt % 20 == 0:
            print('Epoch: {0} - Iter: {1}/{2} - loss: {3:.6f}'.format(
                epoch, itt, len(tr_dataloader), loss.detach().item())
            )

    return avg_loss, net, optimizer


def validate(net, loss_criterion, val_dataloader, epoch, device, is_save=False):
    """Validate the model on the validation set

    Parameters
    ----------
    net : torch.nn.Module
        The network to validate.
    loss_criterion : torch.nn.Module
        The loss criterion to use.
    val_dataloader : torch.utils.data.DataLoader
        The validation data loader.
    epoch : int
        The epoch number.

    Returns
    -------
    avg_loss : float
        The average loss for the epoch.
    avg_ssim : float
        The average SSIM for the epoch.
    """
    avg_loss = 0.0
    avg_ssim = 0.0
    ssim_criterion = SSIM()
    # radial_normalizer = batch_norm(
    #     in_channels = 1,
    # )
    mag = lambda x: (x[..., 0] ** 2 + x[..., 1] ** 2) ** 0.5
    with torch.no_grad():
        for itt, (input, target, name) in enumerate(val_dataloader):
            X = input.to(device)
            y = target.to(device)

            # if config.normalize_input:
            #     X = radial_normalizer(X)
            #     y = radial_normalizer(y)

            y_pred = net(X)
            loss = loss_criterion(y_pred, y)
            ssim = ssim_criterion(mag(y_pred), mag(y))

            avg_loss += loss.detach().item() / len(val_dataloader)
            avg_ssim += ssim.detach().item() / len(val_dataloader)
            
            if itt % 20 == 0:
                print('Epoch: {0} - Iter: {1}/{2} - loss: {3:.6f} - SSIM: {4:.6f}'.format(
                    epoch, itt, len(val_dataloader), loss.detach().item(), ssim.detach().item())
                )

            if is_save:
                if not os.path.isdir(config.models_dir + '/visualize'):
                    os.makedirs(config.models_dir + '/visualize')   
                for i, filename in enumerate(name):
                    out_to_save = mag(y_pred).detach().cpu().numpy()[i, 0, :, :]
                    out_to_save = ((out_to_save - out_to_save.min()) / (out_to_save.max() - out_to_save.min()) * 255).astype(np.uint8)
                    filename = filename.split('/')[-1][:-4]
                    imsave(config.models_dir + '/visualize/' + filename + '.png', out_to_save)
    return avg_loss, avg_ssim


def train(net, optimizer, loss_criterion, tr_dataloader, val_dataloader, device):
    """Train the network

    Parameters
    ----------
    net : torch.nn.Module
        The network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    loss_criterion : torch.nn.Module
        The loss criterion to use.
    tr_dataloader : torch.utils.data.DataLoader
        The training data loader.
    val_dataloader : torch.utils.data.DataLoader
        The validation data loader.
    """
    best_ssim = -1
    for epoch in range(config.num_epochs):
        print(f'Training epoch {epoch}/{config.num_epochs}...')

        optimizer = adjust_learning_rate(epoch, optimizer)

        # Training
        avg_tr_loss, net, optimizer = train_epoch(
            net, optimizer, loss_criterion, tr_dataloader, epoch, device
        )
        print(f'Epoch {epoch} - Avg. training loss: {avg_tr_loss:.3f}')

        # Validation
        avg_vld_loss, avg_vld_ssim = validate(net, loss_criterion, val_dataloader, epoch, device)
        print(f'Epoch {epoch} - Avg. validation loss: {avg_vld_loss:.3f}, SSIM: {avg_vld_ssim:.3f}')


        if (epoch % config.num_epochs_per_saving == 0 or epoch == config.num_epochs - 1) and epoch > 0:
            torch.save({'epoch': epoch, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),},
                os.path.join(config.models_dir, 'checkpoint_epoch_{}.pth'.format(epoch))
            )
            print('Model Saved!')
        if avg_vld_ssim > best_ssim:
            best_ssim = avg_vld_ssim
            print("Best model!")
            torch.save({'epoch': epoch, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),},
                os.path.join(config.models_dir, 'best_model.pth')
            )
            print('Model Saved!')


def save_checkpoint(state, filename='checkpoint.pth'):
    """Save a checkpoint

    Parameters
    ----------
    state : dict
        The state to save.
    is_best : bool
        Whether this is the best model.
    filename : str
        The filename to save the checkpoint to.
    """
    torch.save(state, filename)


def adjust_learning_rate(epoch, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs
    
    Parameters
    ----------
    epoch : int
        The epoch number.
    optimizer : torch.optim.Optimizer
        The optimizer.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The optimizer.
    """
    lr = config.learning_rate * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(222)
    os.makedirs(config.models_dir, exist_ok=True)
    json_path = os.path.join(config.models_dir, 'hyperparameter.json')
    with open(json_path,'w') as f:
        f.write(json.dumps(vars(config), ensure_ascii=False, indent=4, separators=(',', ':')))

    train_dataloader, val_dataloader = get_dataloaders()
    if config.net == "CUNet":
        net = CUNet().to(device)
    if config.net == "RealUNet":
        net = RealUNet().to(device)

    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_criterion = Loss.MSELoss()

    train(net, optimizer, loss_criterion, train_dataloader, val_dataloader, device)
    print("-----Final Test-----")
    net.load_state_dict(torch.load(os.path.join(config.models_dir, 'best_model.pth'), map_location='cpu')["state_dict"])
    avg_vld_loss, avg_vld_ssim = validate(net, loss_criterion, val_dataloader, config.num_epochs, device, is_save=True)
    print(f'Epoch {config.num_epochs} - Avg. validation loss: {avg_vld_loss:.3f}, SSIM: {avg_vld_ssim:.3f}')
