import json

import torch
import torch.nn as nn
import os
import shutil

from simplified_vae.config.config import Config


def save_checkpoint(checkpoint_dir: str,
                    model: nn.Module,
                    optimizer: torch.optim,
                    loss: float,
                    epoch_idx: int,
                    is_best: bool,
                    max_keep: int = 10):

    model_str = "epoch_idx_" + str(epoch_idx) + ".tar"
    torch.save({
        'epoch': epoch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(checkpoint_dir, model_str))

    if is_best:
        shutil.copyfile(os.path.join(checkpoint_dir, model_str), os.path.join(checkpoint_dir, 'model_best.pth.tar'))

    files = sorted(os.listdir(checkpoint_dir))
    rm_files = files[0:max(0, len(files) - max_keep)]
    for f in rm_files:
        os.remove(os.path.join(checkpoint_dir, f))


def load_checkpoint(checkpoint_path: str,
                    model: nn.Module,
                    optimizer=None):

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    if (optimizer is not None):

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, epoch, loss
    else:
        return model, epoch, loss

def write_config(config: Config, logdir: str):

    with open(os.path.join(logdir, 'train_config.json'), 'w') as outfile:
        json.dump(config.training.json(), outfile)
