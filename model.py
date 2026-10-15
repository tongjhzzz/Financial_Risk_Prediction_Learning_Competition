from torch import nn

# TODO 2【nn.Module / 线性层 / 非线性激活 / Sequential】
# 输入 n_features 维，输出 2 维（违约 / 不违约，和视频 CIFAR10 输出 10 维同理）
# 表格数据用不到卷积和池化，用几层 Linear + ReLU 就行，比如 118 -> 256 -> 128 -> 2
# 进阶：每个 Linear 后面加 nn.BatchNorm1d、nn.Dropout(0.3) 试试效果
class LoanNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x