import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import optuna
from optuna.trial import TrialState
from torchvision.datasets import VOCSegmentation  # You can use any dataset, this is just an example.
