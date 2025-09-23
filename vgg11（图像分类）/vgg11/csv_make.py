import csv
import os


def make_csv(img_path):

    """根据图片数据集的路径，生成一个csv文件和一个json文件。"""
    data_name = img_path.split("/")[-1]
    csv_path = os.path.join("./config",f"{data_name}.csv")
    json_path = os.path.join("./config",f"{data_name}.json")
    data_class = os.listdir(img_path)  # 获取所有类别的名称

    class_to_num={}
    for index,class_name in enumerate(data_class): # 遍历每个类别
        class_to_num[class_name] = index

    with open(json_path,'w') as json_file:
        json_file.write(str(class_to_num))
    with open(csv_path,'w',newline='')as csv_file: # 打开csV文件，写入模式
        writer = csv.writer(csv_file)
        for class_name in data_class:
            class_path =os.path.join(img_path, class_name)
            img_name = os.listdir(class_path) # 获取每个类别下的图片名称
            for img in img_name:
                img_path_full = os.path.join(class_path,img) # 完整的图片路径
                label = class_to_num[class_name]
                writer.writerow([img_path_full,label]) #写入csV文件，包含图片路径和标签

if __name__ == "__main__":
    img_path = "./image/cat_dog"  # 替换为你的图片数据集路径
    make_csv(img_path)  # 调用函数生成csv文件和json文件
