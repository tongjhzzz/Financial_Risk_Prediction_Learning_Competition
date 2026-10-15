"""
对应视频「完整的模型验证套路」：加载训练好的模型，对没见过的数据做预测。

视频里：加载模型 -> 预测一张狗的图片 -> 输出类别
这里：  加载模型 -> 预测 testA 的 20 万人 -> 输出违约概率 -> 写成提交用的 csv

运行：uv run python predict.py
产出：submission.csv（格式和 sample_submit.csv 一样：id,isDefault）
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import load_datasets
from model import LoanNet

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_set, val_set, test_set, test_ids = load_datasets()
n_features = test_set[0].shape[0]  # 测试集没有标签，test_set[0] 直接就是 x


# 1. 网络结构：必须和训练时一模一样，否则参数对不上
#    不能写 from train import LoanNet —— 那样会把 train.py 整个执行一遍，重新训练 10 轮
#    所以把模型类单独放进 model.py，train.py 和 predict.py 都从那里 import


# 2. 加载模型：视频里的「方式 2」—— 先建网络，再把保存的参数装进去
#    map_location：不管保存时在哪个设备，先读到 CPU 上，再统一 .to(device)
model = LoanNet(n_features)
model.load_state_dict(torch.load("model_best.pth", map_location="cpu"))
model = model.to(device)

# 3. 预测：和 train.py 的验证步骤几乎一样，只是没有标签、不算 AUC
#    shuffle 必须是 False！否则预测结果的顺序和 id 对不上
test_loader = DataLoader(test_set, batch_size=4096, shuffle=False)

model.eval()
all_prob = []
with torch.no_grad():
    for x in test_loader:          # LoanDataset 在没有 y 时只返回 x
        x = x.to(device)
        outputs = model(x)
        prob = torch.softmax(outputs, dim=1)[:, 1]
        all_prob.append(prob.cpu())
all_prob = torch.cat(all_prob).numpy()

# 4. 写成和 sample_submit.csv 一样的格式：两列 id, isDefault，不要行号
submission = pd.DataFrame({"id": test_ids, "isDefault": all_prob})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"共 {len(submission)} 行，已保存到 submission.csv")
