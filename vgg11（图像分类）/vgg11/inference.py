import torch
from algorithm.VGG11 import VGG11
from algorithm.mobilenetv3 import mobilenetv3
import cv2
from torchvision import transforms


img_path = "./image/cat_dog/cats/cat.12.jpg"
model_path = "./model/mobileney-v3.pth"
device = "cuda"
model = mobilenetv3(n_class=2)
model.load_state_dict(torch.load(model_path))       # 加载模型参数
model = model.to(device)        # 将模型移动到指定设备上进行计算
model.eval()                         # 设置模型为评估模式

img = cv2.imread(img_path)  # 读取图片
img = cv2.resize(img, (224,224))  # 调整图片大小
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
img = transforms.ToTensor()(img)  # 将图片转换为torch数组
img = img.unsqueeze(0)  # 增加一个维度，变成[1,3,224,224]
img = img.to(device)    # 将图片移动到指定设备上进行计算

predict = model(img)  # 将图片输入模型，得到预测结果
predict = torch.argmax(predict, dim=1).item()  # 取最大值的索引
if predict == 0:
    print("预测结果：猫",predict)
else:
    print("预测结果：狗",predict)
