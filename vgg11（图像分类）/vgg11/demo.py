import gradio as gr
import cv2
from algorithm.mobilenetv3 import mobilenetv3
import torch
from torchvision import transforms

# 一：模型必须是评估模式。 二：图像的色彩空间必须和训练时一致。
model_path = "model/mobileney-v3.pth"  # 模型路径
model = mobilenetv3(n_class=2)  # 创建模型实例
json_path = "./config/cat_dog.json"  # 标签文件路径


model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # 加载模型参数
if torch.cuda.is_available():  # 检测是否有可用的GPU
    device = "cuda"
else:
    device = "cpu"
resize = (224, 224)
model = model.to(device)    # 将模型移动到指定设备
model = model.eval()        # 设置模型为评估模式

# 根据我的json数据，将预测结果转为字符串
def num_to_label(json_path, num):
    with open(json_path, 'r', encoding='utf-8') as f:   # 打开json文件
        data = f.read()                 # 读取文件内容
    data = eval(data)  # 将字符串转换为字典
    for key, value in data.items():     # 遍历字典
        if value == num:            # 如果值等于预测数字
            label = key             # 取出对应的键
            break                   # 跳出循环
    return label            # 返回标签字符串

# 推理函数
def inference(img):
    img = cv2.resize(img, (224, 224))  # 调整图像大小
    img = transforms.ToTensor()(img)  # 将图片转换为torch数组
    img = img.unsqueeze(0)                  # 添加批次维度
    img = img.to(device)                    # 将图像移动到指定设备
    with torch.no_grad():                   # 禁用梯度计算
        predict = model(img)                 # 模型推理
    predict = predict.argmax(1).item()      # 获取预测结果
    predict = num_to_label(json_path, predict)  # 将数字标签转换为文本标签
    return str(predict)        # 返回预测结果字符串

# 创建 Gradio 接口
demo = gr.Interface(fn=inference, inputs="image", outputs="text")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8039, share=True)
