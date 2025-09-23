from algorithm.VGG11 import VGG11
from algorithm.mobilenetv3 import mobilenetv3
from data_loader import Mydataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


train_csv = "./config/cat_dog.csv"    # 训练数据的csv表格路径
epoch = 100                 # 训练轮次 ,图像 50 - 100
lr = 0.001                 # 学习率， 0.001 - 0.01
model_save_path = "./model/mobileney-v3.pth"  # 模型保存路径
if torch.cuda.is_available():    # 检测是否显卡可以使用
    device = "cuda"
else:
    device = "cpu"
model = mobilenetv3(n_class = 2)
model.to(device)                # 将模型移动到指定设备上进行计算
dataset = Mydataset(train_csv, (224, 224))  # 创建数据集实例
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)  # 创建数据加载器,返回一个容器

loss_fn = torch.nn.CrossEntropyLoss()       # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # 优化器，使用随机梯度下降法


if __name__ == "__main__":
    for i in range(epoch):   # 训练轮次
        n = 0
        qbar = tqdm(dataloader, desc=f"epoch {i+1}/{epoch}")  # 创建进度条
        for step, (x, y) in enumerate(qbar):    # 遍历数据加载器，x是图片，y是标签，每次读取部分图片
            x = x.to(device)
            y = y.to(device)
            predict = model(x)     # 利用模型计算图像的预测结果
            loss = loss_fn(predict, y)    # 计算损失
            optimizer.zero_grad()           # 清除梯度
            loss.backward()                # 反向传播计算梯度
            optimizer.step()               # 更新参数
            n += 1
            acc = (predict.argmax(1) == y).sum().item() / y.shape[0]  # 计算准确率
            qbar.set_postfix({"损失": loss.item(), "准确率": acc})
        if (i+1) % 10 == 0:  # 每10轮保存一次模型
            torch.save(model.state_dict(), model_save_path)  # 保存模型参数
            print(f"模型已保存到 {model_save_path}")  # 打印保存路径
