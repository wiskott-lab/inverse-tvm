import torch
import random
import numpy as np


def optim_step(optim, loss):
    optim.zero_grad()
    loss.backward()
    optim.step()


def init_seeds(seed):
    seed = random.randint(0, 2147483647) if seed is None else seed  # 32-bit integer
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If you are using CUDA (GPU), set the seed for that as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
