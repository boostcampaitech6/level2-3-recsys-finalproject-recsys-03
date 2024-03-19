import os
import torch
import random
import logging
import numpy as np


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def makedirs(path):
        if not os.path.exists(path):
            os.makedirs(path)


def get_logger(filename):
    logging.basicConfig(
        filename=filename,
        format='%(asctime)s  %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p',
        level=logging.DEBUG,
    )
    logger = logging.getLogger()
    return logger
