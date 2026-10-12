"""
练习：用视频里「完整的模型训练套路」训练贷款违约模型。

数据部分（preprocess.py、dataset.py）已经写好，视频没讲表格数据怎么处理。
下面每个 TODO 都是视频里讲过的内容，按提示补全就能跑。

运行：uv run python train.py
看曲线：uv run tensorboard --logdir logs
"""
import os

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import load_datasets

# 「利用GPU训练（二）」：视频里写 "cuda"，M1 Mac 用 "mps"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_set, val_set, test_set, test_ids = load_datasets()
n_features = train_set[0][0].shape[0]  # 118

# TODO 1【DataLoader 的使用】
# 训练集要打乱，验证集不用。表格数据一行很小，batch_size 可以开大，比如 1024
train_loader = DataLoader(dataset=train_set, batch_size=1024, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=1024, shuffle=False)

# TODO 2【nn.Module / 线性层 / 非线性激活 / Sequential】
# 输入 n_features 维，输出 2 维（违约 / 不违约，和视频 CIFAR10 输出 10 维同理）
# 表格数据用不到卷积和池化，用几层 Linear + ReLU 就行，比如 118 -> 256 -> 128 -> 2
# 进阶：每个 Linear 后面加 nn.BatchNorm1d、nn.Dropout(0.3) 试试效果
class LoanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = LoanNet().to(device)

# TODO 3【损失函数与反向传播 / 优化器】
# 输出 2 维 + 标签是 0/1，所以和视频一样用 CrossEntropyLoss
# 优化器视频用 SGD；这里建议先用 torch.optim.Adam，lr=1e-3（收敛快很多），之后可以换回 SGD 对比
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

writer = SummaryWriter("logs/plain")
os.makedirs("models", exist_ok=True)
epochs = 10
best_auc = 0
totoal_train_step = 0

for epoch in range(epochs):
    print(f"-------第 {epoch + 1} 轮训练开始-------")

    # TODO 4【完整的模型训练套路】训练步骤
    # model.train()，然后对每个 batch：
    #   x, y 放到 device -> 前向 -> 算 loss -> 梯度清零 -> 反向传播 -> 更新参数
    # 每隔一些 step 用 writer.add_scalar("train_loss", ...) 记录一下

    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        loss = loss_fn(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totoal_train_step += 1
        if totoal_train_step % 100 == 0:
            print(f"训练次数: {totoal_train_step}, Loss: {loss.item():.4f}")
            writer.add_scalar("loss", loss.item(), totoal_train_step)

    # TODO 5【完整的模型训练套路】验证步骤
    # model.eval() + with torch.no_grad()，遍历 val_loader
    # 视频里算的是准确率；这里要算 AUC：
    #   概率 = torch.softmax(outputs, dim=1)[:, 1]   # 第 1 列 = 违约概率
    #   把每个 batch 的概率和标签收集起来（记得 .cpu()），最后 roc_auc_score(全部标签, 全部概率)
    # 想想：为什么这里不用准确率？（提示：80% 的人不违约）
    model.eval()
    all_y = []
    prob_y = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            outputs = model(x)
            prob = torch.softmax(outputs, dim=1)[:, 1]
            all_y.append(y)
            prob_y.append(prob.cpu())

        all_y = torch.cat(all_y)
        prob_y = torch.cat(prob_y)
        val_auc = roc_auc_score(all_y, prob_y)
        print(f"验证集AUC: {val_auc:.4f}")

        # TODO 6【网络模型的保存与读取】
        # 用 writer.add_scalar 记录 val_auc
        # 如果 val_auc 比 best_auc 高，就更新 best_auc，并保存到 models/best.pth
        # （推荐视频里的方式 2：只保存 model.state_dict()）
        writer.add_scalar("val_auc", val_auc, epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "model_best.pth")
            print("保存了新的最好的模型")

writer.close()
print(f"最好的验证 AUC：{best_auc:.4f}")
