import torch
import csv
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2


class Mydataset(Dataset):
    def __init__(self, csv_path, resize):
        super(Mydataset, self).__init__()    # 执行父类的初始化方法
        self.csv_path = csv_path
        self.resize = resize
        self.data, self.label = self.csv_loader()  # 调用自己的csv_loader方法读取数据和标签

    def csv_loader(self):     # 根据csv文件读取里面的数据
        data = []       # 存储图片路径
        lable = []      # 存储图片标签
        with open(self.csv_path, 'r') as f:    # 打开csv文件
            reader = csv.reader(f)         # 读取csv文件
            for row in reader:        # 遍历csv文件的每一行
                i, l = row[0], row[1]  # 读取每一行的图片路径和标签
                data.append(i)          # 将图片路径添加到data列表中
                lable.append(int(l))    # 将标签转换为整数并添加到lable列表中
        return data, lable

    def __len__(self):    # 必须写获取你自己的数据集的长度的代码
        return len(self.data)   # 返回data列表的长度

    def __getitem__(self, index):
        """
        index是你的数据集的索引值，你需要根据索引值获取对应的图片和标签，
        并且将图片转换为torch数组，标签转换为torch张量。
        """
        img_path = self.data[index]   # 获取图片路径
        label = self.label[index]     # 获取对应的标签
        img = cv2.imread(img_path)   # 读取图片
        img = cv2.resize(img, self.resize)   # 调整图片大小
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
        img = transforms.ToTensor()(img)  # 将图片转换为torch数组
        label = torch.tensor(label, dtype=torch.long)  # 将标签转换为torch张量
        return img, label  # 返回图片和标签


if __name__ == '__main__':
    dataset = Mydataset('./config/cat_dog.csv', (224, 224))  # 创建数据集实例
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # 创建数据加载器,返回一个容器
    for x, y in dataloader:
        print(x.shape, y)

