import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset/cifar-10-python", train=False, transform=torchvision.transforms.ToTensor(), download=False)

dataloader = DataLoader(dataset, batch_size=64)


class Tudui1(nn.Module):
    def __init__(self):
        super(Tudui1, self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self, x):
            x=self.conv1(x)
            return x


tudui = Tudui1()

writer = SummaryWriter("./logs")

step = 0

for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input",imgs,step)
    output=torch.reshape(output, (-1,3,30,30))
    writer.add_images("output",output,step)
    step=step+1
writer.close()
# 首先卷积来提取特征，但是提取后图片数据量依旧很大，所以需要通过池化来降低特征，从而降低数据量，这样计算量就小了
