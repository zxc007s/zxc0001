import torch
from data_loader import Mydataset
from algorithm.VGG11 import VGG11
from torch.utils.data import DataLoader
from tqdm import tqdm
from algorithm.mobilenetv3 import mobilenetv3

test_csv = r'./config/cat_dog.csv'
model_save_path = r'./model/mobileney-v3.pth'
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model =  mobilenetv3(n_class=2).to(device)
model.load_state_dict(torch.load(model_save_path))
test_db = Mydataset(csv_path=test_csv, resize=(224, 224))
test_loder = DataLoader(test_db, batch_size=100, shuffle=False)

if __name__ == '__main__':
    model.eval()
    correct_total = 0  # 正确预测的总数
    sample_total = 0  # 样本总数

    pbar = tqdm(test_loder, desc="Testing")
    with torch.no_grad():
        for step, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            result = model(x)

            # 计算当前batch的正确预测数和样本数
            batch_correct = (result.argmax(dim=1) == y).sum().item()
            batch_size = x.size(0)  # 使用实际batch大小

            correct_total += batch_correct
            sample_total += batch_size

            # 计算当前准确率
            current_acc = correct_total / sample_total
            pbar.set_postfix({'Acc': f'{current_acc:.4f}',
                              'Samples': f'{sample_total}/{len(test_db)}'})

    final_accuracy = correct_total / sample_total
    print(f"测试集准确率: {final_accuracy:.4f}")
    print(f"正确预测: {correct_total}/{sample_total}")