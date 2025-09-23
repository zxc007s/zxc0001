import torch
from torch import nn


class VGG11(nn.Module):
    def __init__(self, num_class):
        super(VGG11, self).__init__()   # 调用父类的初始化方法
        self.layer1 = nn.Sequential(    # 将多个层组合成一个层
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),  # 输入通道数3，输出通道数64，卷积核大小3x3，填充
            nn.BatchNorm2d(64),                         # 卷积后面必须跟归一化
            nn.ReLU(inplace=True),                      # 归一化后面必须跟激活
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层，池化核大小2x2，步长2
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 全连接层，输入特征数512*7*7，输出特征数4096
            nn.ReLU(inplace=True),                          # 激活函数
            nn.Dropout(p=0.5),            # Dropout层，防止过拟合，丢弃概率0.5
            nn.Linear(4096, 4096),        # 全连接层，输入特征数4096，输出特征数4096
            nn.ReLU(inplace=True),                     # 激活函数
            nn.Dropout(p=0.5),            # Dropout层，防止过拟合，丢弃概率0.5
            nn.Linear(4096, num_class)           # 全连接层，输入特征数4096，输出特征数10（分类数）
        )

    def forward(self, x):     # 输入一个数据，然后根据我们定义好的网络结构进行计算
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)  # 展平操作，将多维张量展平为一维
        out = self.fc(out)  # 全连接层
        return out  # 返回输出结果


if __name__ == '__main__':
    model = VGG11(num_class=2)   # 创建VGG11模型实例，设置分类数为2
    data = torch.randn(1, 3, 224, 224)  # 模拟输入数据，1张图片，3个通道，224x224
    predict = model(data)  # 将输入数据映射为2个类别的预测结果
    print(predict)
