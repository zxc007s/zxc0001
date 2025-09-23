import xml.etree.ElementTree as ET  #
import shutil
import os
from tqdm import tqdm

voc_data_path = "../data/VOC2028"  # VOC数据集路径
yolo_data_path = "../data/yolo_hat"     # YOLO数据集路径
class_names = {
    "hat": 0,
    "person": 1,
}                                   # 类别名称和ID映射

voc_label_path = voc_data_path + "/Annotations"     # VOC标签路径
voc_img_path = voc_data_path + "/JPEGImages"        # VOC图片路径
voc_train_path = voc_data_path + "/ImageSets/Main/train.txt"        # VOC训练集路径
voc_val_path = voc_data_path + "/ImageSets/Main/val.txt"            # VOC验证集路径
voc_test_path = voc_data_path + "/ImageSets/Main/test.txt"          # VOC测试集路径
yolo_train_images = yolo_data_path + "/images/train"                # YOLO训练集图片路径
yolo_val_images = yolo_data_path + "/images/val"                    # YOLO验证集图片路径
yolo_test_images = yolo_data_path + "/images/test"                  # YOLO测试集图片路径
yolo_train_labels = yolo_data_path + "/labels/train"                # YOLO训练集标签路径
yolo_val_labels = yolo_data_path + "/labels/val"                    # YOLO验证集标签路径
yolo_test_labels = yolo_data_path + "/labels/test"                  # YOLO测试集标签路径


def voc_to_yolo(voc_label_path, voc_img_path, voc_split_path, yolo_img_path, yolo_label_path):
    with open(voc_split_path, 'r') as f:            # VOC数据集的训练、验证或测试集文件
        img_names = f.read().strip().split('\n')        # 获取图片名称列表
    for img_name in tqdm(img_names):                    # 遍历每个图片名称
        img_path = os.path.join(voc_img_path, img_name + '.jpg')        # 获取图片路径
        xml_path = os.path.join(voc_label_path, img_name + '.xml')      # 获取XML标签路径
        with open(xml_path, 'r', encoding='utf-8') as f:            # 打开XML文件
            xml_content = f.read()                      # 读取XML内容
            root = ET.fromstring(xml_content)               # 解析XML内容
            width = int(root.find('size/width').text)       # 获取图片宽度
            height = int(root.find('size/height').text)     # 获取图片高度
            yolo_label = []                                 # 初始化YOLO标签列表
            for obj in root.findall('object'):              # 遍历每个目标对象
                class_name = obj.find('name').text          # 获取目标类别名称
                if class_name not in class_names:           # 如果类别名称不在预定义的类别映射中
                    continue                                # 跳过该目标
                class_id = class_names[class_name]          # 获取目标类别ID
                bndbox = obj.find('bndbox')                 # 获取边界框信息
                xmin = int(bndbox.find('xmin').text)        # 获取边界框左上角x坐标
                ymin = int(bndbox.find('ymin').text)        # 获取边界框左上角y坐标
                xmax = int(bndbox.find('xmax').text)        # 获取边界框右下角x坐标
                ymax = int(bndbox.find('ymax').text)        # 获取边界框右下角y坐标
                x_center = (xmin + xmax) / 2 / width        # 计算边界框中心点x坐标
                y_center = (ymin + ymax) / 2 / height       # 计算边界框中心点y坐标
                bbox_width = (xmax - xmin) / width          # 计算边界框宽度
                bbox_height = (ymax - ymin) / height        # 计算边界框高度
                yolo_label.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}") # 将YOLO格式标签添加到列表中
            yolo_img_path_save = os.path.join(yolo_img_path, img_name + '.jpg')     # 保存YOLO格式图片路径
            yolo_label_path_save = os.path.join(yolo_label_path, img_name + ".txt")  #  保存YOLO格式标签路径
            shutil.copy(img_path, yolo_img_path_save)               # 复制图片到YOLO格式图片路径
            with open(yolo_label_path_save, 'w') as label_file:         # 打开YOLO格式标签文件
                label_file.write('\n'.join(yolo_label))             # 将YOLO标签写入文件


if __name__ == "__main__":
    voc_to_yolo(voc_label_path, voc_img_path, voc_train_path, yolo_train_images, yolo_train_labels) # 转换VOC训练集为YOLO格式
    voc_to_yolo(voc_label_path, voc_img_path, voc_val_path, yolo_val_images, yolo_val_labels)       # 转换VOC验证集为YOLO格式
    voc_to_yolo(voc_label_path, voc_img_path, voc_test_path, yolo_test_images, yolo_test_labels)    # 转换VOC测试集为YOLO格式
    print("voc数据集转换为yolo数据集完成")