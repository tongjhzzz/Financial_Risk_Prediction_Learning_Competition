"""
对应视频 P6「Dataset 类代码实战」。

视频里：一个样本 = 一张蚂蚁/蜜蜂图片 + 它的标签
这里：  一个样本 = 一行贷款记录（一个特征向量）+ 是否违约
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class LoanDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X)
        # 测试集没有标签，y=None 时只返回特征
        self.y = None if y is None else torch.from_numpy(y).long()

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


def load_datasets(path="data/processed.npz"):
    data = np.load(path, allow_pickle=True)
    train_set = LoanDataset(data["X_train"], data["y_train"])
    val_set = LoanDataset(data["X_val"], data["y_val"])
    test_set = LoanDataset(data["X_test"])
    return train_set, val_set, test_set, data["test_ids"]


if __name__ == "__main__":
    train_set, val_set, test_set, _ = load_datasets()
    x, y = train_set[0]
    print(len(train_set), len(val_set), len(test_set))
    print(x.shape, x.dtype, y)
