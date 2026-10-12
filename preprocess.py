"""
把原始 CSV 变成神经网络能吃的纯数字矩阵 —— 相当于视频里 transforms 那一步。

视频里图片本来就是数字，只要 ToTensor + Normalize。
表格数据麻烦一些，要先处理：字符串、缺失值、异常值、类别特征，最后才是 Normalize。

运行：uv run python preprocess.py
产出：data/processed.npz（训练集 / 验证集 / 测试集的特征矩阵）
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# 类别不多、数字本身没有大小意义的列 -> one-hot
ONEHOT_COLS = ["homeOwnership", "verificationStatus", "purpose", "regionCode",
               "initialListStatus", "applicationType", "term"]
# 类别特别多的列 -> 频数编码（这个值在数据里出现了多少次）
FREQ_COLS = ["employmentTitle", "title", "postCode"]
# 长尾分布（少数人数值特别大）-> 先取 log 再标准化
LOG_COLS = ["loanAmnt", "installment", "annualIncome", "revolBal"]
# 没用的列：id、常数列、和别的列完全重复的列
DROP_COLS = ["id", "policyCode", "n3", "ficoRangeHigh"]
N_COLS = [f"n{i}" for i in range(15) if i != 3]


def clean(df):
    """把字符串变数字，构造几个新特征。训练集和测试集用同一套规则。"""
    df = df.copy()

    # grade A~G、subGrade A1~G5：有先后顺序，直接映射成 0,1,2...
    df["grade"] = df["grade"].map({g: i for i, g in enumerate("ABCDEFG")})
    df["subGrade"] = df["subGrade"].map(
        {f"{g}{n}": i * 5 + n - 1 for i, g in enumerate("ABCDEFG") for n in range(1, 6)})

    # "10+ years" -> 10，"< 1 year" -> 0，"5 years" -> 5，缺失保持 NaN
    df["employmentLength"] = (df["employmentLength"]
                              .replace({"10+ years": "10", "< 1 year": "0"})
                              .str.extract(r"(\d+)", expand=False).astype(float))

    # 日期 -> 数字：放款日期距 2007-06 有几个月；信用历史有几个月
    issue = pd.to_datetime(df["issueDate"])
    earliest = pd.to_datetime(df["earliesCreditLine"], format="%b-%Y")
    df["issueMonths"] = (issue.dt.year - 2007) * 12 + issue.dt.month - 6
    df["creditHistoryMonths"] = ((issue.dt.year - earliest.dt.year) * 12
                                 + issue.dt.month - earliest.dt.month)
    df = df.drop(columns=["issueDate", "earliesCreditLine"])

    # 匿名特征 n0~n14 基本是同一批人一起缺失，"缺失"本身可能就是信息
    df["n_missing"] = df["n0"].isna().astype(float)

    # 几个比较直观的比例特征
    df["loanToIncome"] = df["loanAmnt"] / (df["annualIncome"] + 1)
    df["installmentToIncome"] = df["installment"] * 12 / (df["annualIncome"] + 1)

    return df.drop(columns=DROP_COLS)


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("testA.csv")
    test_ids = test["id"].to_numpy()
    y = train.pop("isDefault").to_numpy()

    # 训练集和测试集拼在一起处理，保证 one-hot 后两边的列完全一致
    n_train = len(train)
    full = clean(pd.concat([train, test], ignore_index=True))

    for c in FREQ_COLS:
        full[c] = np.log1p(full[c].map(full[c].value_counts())).fillna(0)
    for c in LOG_COLS:
        full[c] = np.log1p(full[c])
    full = pd.get_dummies(full, columns=ONEHOT_COLS, dtype=float)

    X_all, X_test = full.iloc[:n_train], full.iloc[n_train:]

    # 切出 20% 当验证集（视频里 CIFAR10 自带 train/test，这里要自己切）
    # stratify=y：两边违约比例都保持 20%
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y, test_size=0.2, random_state=SEED, stratify=y)

    # 下面这些统计量（分位数、中位数、均值、标准差）只能从训练部分算，
    # 再原样用到验证集和测试集上，否则验证分数会偏乐观（数据泄露）
    lo, hi = X_train.quantile(0.001), X_train.quantile(0.999)
    median = X_train.median()

    def clip_fill(X):
        X = X.clip(lo, hi, axis=1)  # 截掉极端异常值，比如 dti=999
        return X.fillna(median)     # 缺失值用中位数补

    X_train, X_val, X_test = clip_fill(X_train), clip_fill(X_val), clip_fill(X_test)

    # 就是 transforms.Normalize(mean, std)，只不过 mean/std 是每一列各算一个
    mean, std = X_train.mean(), X_train.std().replace(0, 1)
    X_train, X_val, X_test = [((X - mean) / std).to_numpy(np.float32)
                              for X in (X_train, X_val, X_test)]

    os.makedirs("data", exist_ok=True)
    np.savez_compressed("data/processed.npz",
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, test_ids=test_ids,
                        feature_names=np.array(full.columns))
    print(f"特征数 {X_train.shape[1]}；训练 {len(X_train)}，验证 {len(X_val)}，测试 {len(X_test)}")
    print(f"违约比例：训练 {y_train.mean():.4f}，验证 {y_val.mean():.4f}")


if __name__ == "__main__":
    main()
