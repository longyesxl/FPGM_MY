import vgg
import torch

hh=vgg.Conv2d_Mask(3, 3, kernel_size=3, padding=1).cuda()
hh.Conv2d.weight.data[0]=1
hh.Conv2d.weight.data[1]=2
hh.Conv2d.weight.data[2]=3
print(hh.mask)
hh.mask[0]=1
hh.mask[1]=0
hh.mask[2]=1
print(hh.Conv2d.weight.data)
print(hh.mask)
nonz_index=torch.nonzero(hh.mask.view(-1)).view(-1)
nonz_nub=len(nonz_index)
conv2d_w=torch.ones(hh.Conv2d.weight.data.size()).copy_(hh.Conv2d.weight.data)[nonz_index]
conv2d_b=torch.ones(hh.Conv2d.bias.data.size()).copy_(hh.Conv2d.bias.data)[nonz_index]
gg=vgg.Conv2d_Mask(3, 2, kernel_size=3, padding=1).cuda()
gg.Conv2d.weight.data.copy_(conv2d_w)
gg.Conv2d.bias.data.copy_(conv2d_b)
print(conv2d_w)
print(conv2d_b)
tryit=torch.ones(1,3,3,3).cuda()
rz=hh(tryit)
print(rz)
rz2=gg(tryit)
print(rz2)