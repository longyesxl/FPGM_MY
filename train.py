import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import vgg

transform_train =transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

vggnet=vgg.vgg19()

criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(vggnet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) 

testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True) 
if __name__ == "__main__":
    for epoch in range(10):
        print('\nEpoch: %d' % (epoch + 1))
        vggnet.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = inputs, labels
            optimizer.zero_grad()
            outputs = vggnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
        with torch.no_grad():
            correct = 0
            total = 0
            for (images, labels) in testloader:
                vggnet.eval()
                images, labels = images, labels
                outputs = vggnet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = 100. * correct / total
            torch.save(vggnet.state_dict(), 'model/net_%03d.pth' % (epoch + 1))
            print("Training Finished, TotalEPOCH=%d" % epoch)