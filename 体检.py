"""
体检.py —— 拿到任何一张新表，先跑它，看清楚手里有什么。

用法：
    uv run python 体检.py train.csv
    uv run python 体检.py practice/adult.data --no-header --names-file practice/adult_列名.txt

它只做诊断，不改数据、不写文件。输出三部分：
    1. 全局概况（多少行多少列、目标列情况）
    2. 逐列体检表（类型、缺失、取值个数、样例、脚本的猜测）
    3. 需要你人工判断的列，以及脏数据警告

注意：脚本的"猜测"只是起点，不是答案。
      比如 regionCode 是 1~50 的整数，脚本猜"数值·离散"，
      但它到底是"有顺序"还是"纯类别"，只有了解业务才知道。
"""
import argparse

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 28)


def guess_type(s, n_rows):
    """看一眼这一列，猜它属于哪种类型。"""
    nunique = s.nunique(dropna=True)
    n_missing = s.isna().sum()
    n_valid = n_rows - n_missing

    if nunique == 0:
        return "全空", "整列都是缺失值，直接删"
    if nunique == 1:
        return "常数", "整列一个值，没有任何信息，删掉"
    if n_valid > 0 and nunique == n_valid and n_rows > 100:
        return "疑似每行唯一", "像 id / 编号，删掉（除非它有业务含义）"

    # 能解析成日期的文字列
    if s.dtype == object or str(s.dtype) == "str":
        sample = s.dropna().head(200)
        if len(sample):
            converted = pd.to_datetime(sample, errors="coerce", format="mixed")
            if converted.notna().mean() > 0.9:
                return "日期", "转成日期，再拆出年 / 月 / 星期，或算时间差"

    if pd.api.types.is_numeric_dtype(s):
        if nunique <= 15:
            return "数值·离散", "取值少：可能是类别（看有没有顺序），也可能是计数"
        if nunique <= 100 and float(nunique) / max(n_valid, 1) < 0.05:
            return "数值·离散(偏类别)", "取值不多，先看分布再决定"
        skew = s.skew() if n_valid > 2 else 0
        if abs(skew) > 3:
            return "数值·连续(长尾)", "少数值特别大，考虑 np.log1p() 压一压"
        return "数值·连续", "看异常值，考虑 clip"

    # 剩下的按文字处理
    if nunique <= 2:
        return "文本·二分类", "只有两种取值，映射成 0/1"
    if nunique <= 15:
        return "文本·低基数类别", "one-hot（pd.get_dummies）；有顺序的话改用 map"
    if nunique <= max(n_rows * 0.05, 50):
        return "文本·中基数类别", "位数偏多，one-hot 会撑大表；可先试 one-hot + 频数编码"
    if nunique >= n_valid * 0.5:
        return "文本·近乎唯一", "像自由文本或编号，多半要删或做特殊处理"
    return "文本·高基数类别", "别 one-hot，用频数编码 (value_counts + map)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="csv 文件路径")
    p.add_argument("--no-header", action="store_true", help="文件没有表头")
    p.add_argument("--names-file", help="没有表头时，每行一个列名的文本文件")
    p.add_argument("--sep", default=",", help="分隔符，默认逗号")
    p.add_argument("--target", help="目标列的名字（有表头时用）")
    p.add_argument("--nrows", type=int, default=None, help="只读前 N 行，文件很大时用")
    args = p.parse_args()

    names = None
    if args.names_file:
        with open(args.names_file, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    elif args.no_header:
        # 没有列名就临时造几个，方便先看一眼
        head = pd.read_csv(args.path, sep=args.sep, header=None, nrows=1)
        names = [f"第{i + 1}列" for i in range(head.shape[1])]

    if args.no_header:
        df = pd.read_csv(args.path, sep=args.sep, header=None, names=names, nrows=args.nrows)
    else:
        df = pd.read_csv(args.path, sep=args.sep, nrows=args.nrows)

    n_rows, n_cols = df.shape

    print("=" * 100)
    print(f"【1】全局概况   文件：{args.path}")
    print("=" * 100)
    print(f"行数 {n_rows}    列数 {n_cols}    占用内存 {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print()

    if args.target and args.target in df.columns:
        t = df[args.target]
        print(f"目标列 {args.target}")
        print(t.value_counts(dropna=False).head(10).to_string())
        if t.nunique() == 2:
            print(f"→ 二分类，正例（取值小的那个通常在后面）比例：{t.value_counts(normalize=True).min():.4f}")
            print("→ 指标优先考虑 AUC；如果比例悬殊（比如 1:10），准确率就没意义了")
        print()

    # ---------- 逐列体检 ----------
    print("=" * 100)
    print("【2】逐列体检表")
    print("=" * 100)
    rows = []
    for c in df.columns:
        s = df[c]
        kind, advice = guess_type(s, n_rows)
        sample = s.dropna().unique()[:3]
        rows.append({
            "列名": c,
            "dtype": str(s.dtype),
            "缺失%": round(s.isna().mean() * 100, 2),
            "取值数": s.nunique(dropna=True),
            "样例": " | ".join(str(v)[:12] for v in sample),
            "脚本猜测": kind,
            "建议": advice,
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    # ---------- 脏数据检查 ----------
    print("=" * 100)
    print("【3】脏数据检查")
    print("=" * 100)
    problems = 0

    # 前后有空格
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            s = df[c].dropna()
            if len(s) and (s != s.str.strip()).any():
                n = (s != s.str.strip()).sum()
                print(f"⚠  {c}：有 {n} 个取值前后带空格，比如 {s[s != s.str.strip()].iloc[0]!r}")
                print("     → 读取时加 skipinitialspace=True，或对整列做 .str.strip()")
                problems += 1

    # 缺失值伪装（先 strip 再比，因为真实数据里经常带空格，比如 " ?"）
    FAKE_NA = ["?", "-", "--", "N/A", "NA", "null", "NULL", "None", "none",
               "unknown", "Unknown", "", " ", "."]
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            s = df[c].dropna().astype(str)
            if not len(s):
                continue
            stripped = s.str.strip()
            hit = stripped.isin(FAKE_NA)
            if hit.any():
                vals = stripped[hit].unique()[:3]
                print(f"⚠  {c}：发现 {int(hit.sum())} 个伪装成文字的缺失值，"
                      f"比如 {list(vals)}（占 {hit.mean() * 100:.1f}%）")
                print("     → 先用 .replace({'?': np.nan}) 之类换成真正的空值，再统一处理")
                print("     → 注意：'缺失%' 那一列只统计真正的空值，这里的不算在内")
                problems += 1

    # 常数列
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const:
        print(f"⚠  常数列（只有一个值，无信息）：{const}")
        problems += 1

    # 完全重复的列
    dup_pairs = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                dup_pairs.append((cols[j], cols[i]))
    if dup_pairs:
        print(f"⚠  完全重复的列（后者可以删）：{dup_pairs}")
        problems += 1

    # 高缺失列
    high_missing = [c for c in df.columns if df[c].isna().mean() > 0.4]
    if high_missing:
        print(f"⚠  缺失超过 40% 的列：{high_missing}  → 考虑删掉，或只保留'是否缺失'标记")
        problems += 1

    # 缺失成组出现
    missing_cols = [c for c in df.columns if 0 < df[c].isna().mean() < 0.4]
    if len(missing_cols) >= 3:
        flags = df[missing_cols].isna()
        same = flags.eq(flags.iloc[:, 0], axis=0).all().all()
        if same:
            print(f"⚠  这些列 {missing_cols} 的缺失位置完全相同")
            print("     → 它们很可能是同一批样本一起缺的，加一列'是否缺失'就够了")

    if problems == 0:
        print("没发现明显问题。")
    print()

    # ---------- 需要人工判断的 ----------
    print("=" * 100)
    print("【4】脚本拿不准、必须你自己判断的")
    print("=" * 100)
    print("脚本只能看数据的'形状'，看不出'含义'。下面这些必须你来定：")
    print()
    print("  a) 哪些是 ID / 流水号？脚本猜不准，但业务上你知道它没用 → 删")
    print("  b) 哪些整数其实是有顺序的等级（比如信用分档 1~7）？")
    print("     判断方法：算一下每一档的违约率，如果单调递增或递减 → 有顺序，用 map 保留")
    print("     如果上下起伏没有规律 → 当纯类别，用 one-hot")
    print("  c) 哪些列在业务上是'泄露'信息？比如放款后的还款记录，预测时根本拿不到 → 删")
    print("  d) 文本列里是不是藏着可以拆出来的信息？比如 '5 years' 能抠出 5，")
    print("     'Aug-2001' 能算成距今天数，'3室2厅' 能拆成 3 和 2")
    print("  e) 数值列能不能两两相除造出新特征？（贷款额 / 收入 比单看两个数有用）")
    print()


if __name__ == "__main__":
    main()
