import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import vgg
import network

prunit=network.GALPRUN(128,"../input/vggnet_030.pth")
prunit.train(30)