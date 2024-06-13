import argparse
import os.path

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#from models import EnDecoder, DuaLGR, GNN
from evaluation import eva
import matplotlib.pyplot as plt
#from visulization import plot_loss, plot_tsne
import pandas as pd

import scipy.io

mat = scipy.io.loadmat('ACM3025_results.mat')

y = np.squeeze(mat['gt'])
pre_y = np.squeeze(mat['pre_gt'])

# print("y:", y)
# print("pre_y:", pre_y)
# exit(0)

print('---------------------------------------')
acc, nmi, ari, f1 = eva(y, pre_y, 'Final epoch')