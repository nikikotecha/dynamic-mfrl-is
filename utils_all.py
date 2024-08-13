import os
import random
import time

import numpy as np
import torch
import torch.nn as nn


def get_current_time_tag():
    """
    Return string of current time.

    Returns
    -------
    time_tag: str
    """
    time_tag = "_"+time.strftime('%y%m%d_%H%M', time.localtime(time.time()))
    return time_tag


def make_setting_txt(args, path):
    """
    Save current setting(args) to txt for easy check.

    Parameters
    ----------
    args: argparse.Namespace
        args which contains current setting.
    path: str
        Path where txt file is stored.
    """
    txt_path = os.path.join(path, 'args.txt')
    f = open(txt_path, 'w')
    for arg in vars(args):
        content = arg + ': ' + str(getattr(args, arg)) + '\n'
        f.write(content)
    f.close()


def set_random_seed(rand_seed):
    """
    Set random seeds.
    We might use np.random.RandomState() to update this function.

    Parameters
    ----------
    rand_seed: int
    """
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)


def init_weights(m):
    """
    Define the initialization function for the layers.

    Parameters
    ----------
    m
        Type of the layer.
    """
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def validate_setting(args):
    """
    Validate the current setting(=args).

    Parameters
    ----------
    args: argparse.Namespace
    """
    if args.mode_ac:
        assert len(args.h_dims_a) != 0 and args.lr_a != 0, "Error: actor setting."
    if args.mode_psi:
        assert len(args.h_dims_p) != 0 and args.lr_p != 0, "Error: psi setting."
    else:
        assert len(args.h_dims_c) != 0 and args.lr_c != 0, "Error: critic setting."
    if args.mode_reuse_networks:
        dict_trained = torch.load(args.file_path)
        args_trained = dict_trained['args']
        is_true = (args.mode_psi == args_trained.mode_psi) and (args.mode_ac == args_trained.mode_ac)
        assert is_true, "You can not reuse other networks which modes are not matched."
    #assert args.num_types == len(args.num_agents), "Error: number of types." # Not necessary for now.