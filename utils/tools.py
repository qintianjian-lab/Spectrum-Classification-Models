import random
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch


def check_model_name(model_name: str):
    """
    Check model_name is in implemented_model_list
    :param model_name:
    :return:
    """
    implemented_model_list = ['sscnn', 'cnet', 'rcnet', 'racnet', 'convnext1d', 'onedimcnn']
    assert model_name in implemented_model_list, f'[Error] model_name must be in {implemented_model_list}'


def set_random_seed(random_seed: Union[float, int]):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    pl.seed_everything(random_seed)
