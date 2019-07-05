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
    def init_conv2d_distance_rate(self,distance_rate=0.1):
        for layer in self.vggnet.features:
            if isinstance(layer, vgg.Conv2d_Mask):
                layer.distance_rate=distance_rate
    def init_linear_distance_rate(self,distance_rate=0.1):
        for layer in self.vggnet.classifier:
            if isinstance(layer, vgg.Linear_Mask):
                layer.distance_rate=distance_rate
    def make_model_list(self):
        model_list_f=[]
        model_list_c=[]
        in_channels=[0,1,2]
        model_list_f_w=[]
        model_list_c_w=[]
        bn=False
        for layer in self.vggnet.features:
            if isinstance(layer, vgg.Conv2d_Mask):
                nonz_index=torch.nonzero(layer.mask.view(-1)).view(-1)
                nonz_nub=len(nonz_index)
                conv2d_w=torch.ones(layer.Conv2d.weight.data.size()).copy_(layer.Conv2d.weight.data)[nonz_index][:,in_channels]
                conv2d_b=torch.ones(layer.Conv2d.bias.data.size()).copy_(layer.Conv2d.bias.data)[nonz_index]
                model_list_f.append(nonz_nub)
                model_list_f_w.append([conv2d_w,conv2d_b])
                in_channels=nonz_index
            if isinstance(layer, nn.BatchNorm2d):
                bn=True
            if isinstance(layer, nn.MaxPool2d):
                model_list_f.append("M")
        for layer in self.vggnet.classifier:
            if isinstance(layer, vgg.Linear_Mask):
                nonz_index=torch.nonzero(layer.mask.view(-1)).view(-1)
                nonz_nub=len(nonz_index)
                linear_w=torch.ones(layer.Linear.weight.data.size()).copy_(layer.Linear.weight.data)[nonz_index][:,in_channels]
                linear_b=torch.ones(layer.Linear.bias.data.size()).copy_(layer.Linear.bias.data)[nonz_index]
                model_list_c.append(nonz_nub)
                model_list_c_w.append([linear_w,linear_b])
                in_channels=nonz_index
            if isinstance(layer, nn.Linear):
                linear_w=layer.weight.data[:,in_channels]
                linear_b=layer.bias.data
                model_list_c.append(len(linear_b))
                model_list_c_w.append([linear_w,linear_b])

        return model_list_f,model_list_c,bn,model_list_f_w,model_list_c_w
    def prunself(self):
        mod=make_model(self,is_mask=True)
        self.vggnet=mod
    def make_model(self,is_mask=True):
        model_list_f,model_list_c,bn,model_list_f_w,model_list_c_w=self.make_model_list()
        print(len(model_list_c))
        features=vgg.make_layers(model_list_f,bn,is_mask=is_mask)
        model_out=vgg.VGG(features,num_classes=10, init_weights=False,cl_n1=model_list_f[-2],cl_n2=model_list_c[0],cl_n3=model_list_c[1],is_mask=is_mask)
        ii=0
        for layer in model_out.features:
            if is_mask and isinstance(layer, vgg.Conv2d_Mask):
                layer.Conv2d.weight.data.copy_(model_list_f_w[ii][0])
                layer.Conv2d.bias.data.copy_(model_list_f_w[ii][1])
                ii+=1
            if is_mask==False and isinstance(layer, nn.Conv2d):
                layer.weight.data.copy_(model_list_f_w[ii][0])
                layer.bias.data.copy_(model_list_f_w[ii][1])
                ii+=1
        ii=0
        for layer in model_out.classifier:
            if is_mask and isinstance(layer, vgg.Linear_Mask):
                layer.Linear.weight.data.copy_(model_list_c_w[ii][0])
                layer.Linear.bias.data.copy_(model_list_c_w[ii][1])
                ii+=1
            if is_mask and isinstance(layer, nn.Linear):
                layer.weight.data.copy_(model_list_c_w[ii][0])
                layer.bias.data.copy_(model_list_c_w[ii][1])
                ii+=1
                print(ii)
            if is_mask==False and isinstance(layer, nn.Linear):
                layer.weight.data.copy_(model_list_c_w[ii][0])
                layer.bias.data.copy_(model_list_c_w[ii][1])
                ii+=1
        return model_out
    def change_mask(self):
        nub_f_pruned=0.0
        nub_f_all=0.0
        nub_c_pruned=0.0
        nub_c_all=0.0
        for layer in self.vggnet.features:
            if isinstance(layer, vgg.Conv2d_Mask):
                weight_torch= torch.ones(layer.Conv2d.weight.data.size()).copy_(layer.Conv2d.weight.data)
                similar_pruned_num = int(weight_torch.size()[0] * layer.distance_rate)
                weight_vec = weight_torch.view(weight_torch.size()[0], -1).numpy()
                similar_matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
                similar_sum = np.sum(similar_matrix, axis=0)
                similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
                zz=torch.ones_like(layer.mask.data).cpu()
                for si_index in similar_small_index:
                    zz[si_index,0,0]=0
                layer.mask.data.copy_(zz)
                nub_f_pruned+=(layer.mask.size()[0]-len(torch.nonzero(layer.mask)))
                nub_f_all+=layer.mask.size()[0]
        for layer in self.vggnet.classifier:
            if isinstance(layer, vgg.Linear_Mask):
                weight_torch= torch.ones(layer.Linear.weight.data.size()).copy_(layer.Linear.weight.data)

                similar_pruned_num = int(weight_torch.size()[0] * layer.distance_rate)
                weight_vec = weight_torch.view(weight_torch.size()[0], -1).numpy()
                similar_matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
                similar_sum = np.sum(similar_matrix, axis=0)
                similar_small_index = similar_sum.argsort()[: similar_pruned_num]
                zz=torch.ones(layer.mask.data.size())
                for si_index in similar_small_index:
                    print(si_index)
                    zz[si_index]=0
                layer.mask.data.copy_(zz)
                nub_c_pruned+=(layer.mask.size()[0]-len(torch.nonzero(layer.mask)))
                nub_c_all+=layer.mask.size()[0]
        print(nub_f_pruned/nub_f_all)
        print(nub_c_pruned/nub_c_all)

    def train(self,epoch_time, lr=0.001,momentum=0.9, weight_decay=5e-4,distance_rate=0.1,train_add=False,distance_rate_add=0.01,distance_rate_mul=0.1,distance_rate_time=4,train_conv=True,train_linear=False):
        self.optimizer = optim.SGD(self.vggnet.parameters(), lr=lr,momentum=momentum, weight_decay=weight_decay)
        i_dis_t=0
        for epoch in range(epoch_time):
            st=time.time()
            i_dis_t+=1
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
            if train_conv:   
                self.init_conv2d_distance_rate(distance_rate)
            if train_linear:   
                self.init_linear_distance_rate(distance_rate)
            self.change_mask()
            if(i_dis_t>=distance_rate_time):
                distance_rate+=distance_rate_add
                i_dis_t=0
            print("Training Finished, TotalEPOCH=%d,Epochtime=%d" % (epoch,ed-st))
                
            self.init_linear_distance_rate(distance_rate)
            self.change_mask()
            if(i_dis_t>=distance_rate_time):
                if train_add:
                    distance_rate+=distance_rate_add
                else:
                    distance_rate=1.0-(1.0-distance_rate)*(1.0-distance_rate_mul)
                i_dis_t=0
            print("Training Finished, TotalEPOCH=%d,Epochtime=%d" % (epoch,ed-st))