import vgg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from scipy.spatial import distance
import numpy as np
class GALPRUN():
    def __init__(self,batchSize,vgg_path=""):    
        transform_train =transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
        self.vggnet=vgg.vgg16_bn().cuda()
        if(vgg_path!=""):
            self.vggnet.load_state_dict(torch.load(vgg_path))
        self.optimizer = optim.SGD(self.vggnet.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
        self.trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train) 
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batchSize, shuffle=True) 
        self.testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=256, shuffle=True) 
        self.criterion = nn.CrossEntropyLoss()  
        self.change_mask()
    def change_mask(self,distance_rate=0.1):
        for layer in self.vggnet.features:
            if isinstance(layer, vgg.Conv2d_Mask):
                weight_torch=layer.Conv2d.weight.data
                similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
                weight_vec = weight_torch.view(weight_torch.size()[0], -1).cpu().numpy()
                similar_matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
                similar_sum = np.sum(similar_matrix, axis=0)
                similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
                for si_index in similar_small_index:
                    layer.mask.data[si_index,0,0]=0
                #print(len(torch.nonzero(layer.mask)))


    def train(self,epoch_time, lr=0.01,momentum=0.9, weight_decay=5e-4):
        self.optimizer = optim.SGD(self.vggnet.parameters(), lr=lr,momentum=momentum, weight_decay=weight_decay)

        for epoch in range(epoch_time):
            st=time.time()
            print('\nEpoch: %d' % (epoch + 1))
            self.vggnet.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, (inputs, labels) in enumerate(self.trainloader, 0):
                length = len(self.trainloader)
                inputs, labels = inputs.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                outputs = self.vggnet(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d] Loss: %.03f | Acc: %.3f%% '
                            % (epoch + 1,sum_loss / (i + 1), 100. * correct / total))
            with torch.no_grad():
                correct = 0
                total = 0
                for (images, labels) in self.testloader:
                    self.vggnet.eval()
                    images, labels = images.cuda(), labels.cuda()
                    outputs = self.vggnet(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                acc = 100. * correct / total
                torch.save(self.vggnet.state_dict(), 'model/vggnet_%03d.pth' % (epoch + 1))
                ed=time.time()
            self.change_mask()
            print("Training Finished, TotalEPOCH=%d,Epochtime=%d" % (epoch,ed-st))

GALPRUN(128)