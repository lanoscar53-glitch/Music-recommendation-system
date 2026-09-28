# -*- coding: utf-8 -*-
"""
=============================================================================
音乐推荐系统 v5: 核心算法全部手写 numpy 实现
=============================================================================
数据: YearPredictionMSD 本地 CSV -> D:\\msd_data\\yearpredictionmsd.csv
      515,345 首歌 × 91 列 (year + 12 timbre均值 + 78 timbre协方差)

与上一版的区别:
    - SVM 替换为【多项 Logistic 回归】(Softmax 回归, 手写 numpy)
    - PCA 用【SVD 分解】手写实现 (np.linalg.svd)
    - KMeans 用【k-means++ 初始化 +  Lloyd 迭代】手写实现
    - 300 个随机用户 (无种子)

三个核心数学流程 (代码中有逐步推导注释):
    1) PCA-SVD:  X 中心化 -> X = UΣVᵀ -> 主成分 = V 的前 d 列
    2) KMeans:   k-means++ 选初始质心 -> 交替执行
                 "按最近质心分配簇" 与 "质心=簇内均值" 直到收敛
    3) Softmax 回归: z = XW+b -> p = softmax(z)
                 -> 交叉熵损失 L = -Σy·log(p) + λ||W||²
                 -> 梯度 ∂L/∂W = Xᵀ(p-y)/n + 2λW -> 梯度下降

用法:
    python music_recommender_numpy.py                 # 默认 15 万首, 300 用户
    python music_recommender_numpy.py --users 100     # 快速预览
=============================================================================
"""

import argparse
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLUMNS = (["year"]
           + [f"timbre_avg_{i}" for i in range(1, 13)]
           + [f"timbre_cov_{i}" for i in range(1, 79)])
DATA_PATH = r"D:\msd_data\yearpredictionmsd.csv"

TIMBRE_MEANING = {"timbre_avg_1": "响度", "timbre_avg_2": "明亮度",
                  "timbre_avg_3": "打击感", "timbre_avg_4": "起音锐度"}


# ================================================================ 1. 加载
def load_data(path, max_rows):
    print(f"[加载] {path} ({os.path.getsize(path) / 1024**2:.0f} MB) ...")
    df = pd.read_csv(path, header=None, names=COLUMNS,
                     nrows=max_rows, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    print(f"[加载] 共 {len(df)} 首歌")
    return df


# ================================================================ 2. 手写 PCA (SVD 分解)
def my_pca(X, n_components):
    """
    PCA 的 SVD 实现 (数学流程):

    1. 中心化:  Xc = X - mean(X)          # 每个特征减去均值
    2. SVD 分解: Xc = U Σ Vᵀ
       - Σ 的对角线 σ_i 是奇异值, 从大到小排列
       - V 的第 i 列 v_i 是第 i 个主成分方向 (特征空间中的单位向量)
    3. 解释方差: 第 i 个主成分的方差 = σ_i² / (n-1)
       解释方差比 = σ_i² / Σσ_j²
    4. 投影: 降维后的坐标 = Xc @ V[:, :d]
       即把每首歌投影到前 d 个主成分方向上

    返回 (投影坐标 Xp, 主成分矩阵 V, 解释方差比)
    """
    Xc = X - X.mean(axis=0)                      # 1. 中心化
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # 2. SVD
    var_ratio = (S ** 2) / (S ** 2).sum()        # 3. 解释方差比
    V = Vt.T                                     # V 的列 = 主成分方向
    Xp = Xc @ V[:, :n_components]                # 4. 投影到前 d 维
    return Xp, V, var_ratio


# ================================================================ 3. 手写 KMeans
def _kmeans_plusplus(X, k, rng):
    """
    k-means++ 初始化: 让初始质心彼此远离, 避免陷入差的局部最优。
    第 1 个质心随机选; 之后每个点被选中的概率正比于
    "它到最近已有质心的距离平方" D(x)² —— 离得越远越可能被选中。
    """
    n = len(X)
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        # 每个点到最近质心的距离平方: ||x-c||² = |x|²+|c|²-2x·c
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / d2.sum()
        centers.append(X[rng.choice(n, p=probs)])
    return np.array(centers)


def _pairwise_dist2(X, C):
    """
    样本-质心距离矩阵 (向量化):
    ||x - c||² = ||x||² + ||c||² - 2·x·c
    用矩阵乘法一次性算出 n×k 个距离, 避免 Python 双重循环。
    """
    return (np.sum(X ** 2, axis=1, keepdims=True)
            + np.sum(C ** 2, axis=1) - 2 * X @ C.T)


def my_kmeans(X, k, n_init=5, max_iter=100, seed=42):
    """
    KMeans 的 Lloyd 迭代 (数学流程):

    重复以下两步直到质心不再变化 (或达到 max_iter):
      分配步: 每个点分给距离最近的质心  c_i = argmin_j ||x_i - μ_j||²
      更新步: 每个质心移到其簇内所有点的均值  μ_j = mean({x_i: c_i=j})

    目标函数 (惯性 inertia) = Σ||x_i - μ_{c_i}||², 每轮迭代单调不增。
    做 n_init 次不同初始化, 保留 inertia 最小的一组结果。

    返回 (每个点的簇标签, 质心, 惯性)
    """
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        C = _kmeans_plusplus(X, k, rng)
        for _ in range(max_iter):
            labels = np.argmin(_pairwise_dist2(X, C), axis=1)   # 分配步
            # 更新步: 用 bincount 快速求各簇均值 (空簇保留原质心)
            sums = np.zeros_like(C)
            np.add.at(sums, labels, X)
            cnt = np.bincount(labels, minlength=k)
            new_C = np.where(cnt[:, None] > 0,
                             sums / np.maximum(cnt, 1)[:, None], C)
            if np.allclose(new_C, C):                            # 收敛
                break
            C = new_C
        inertia = float(np.sum(np.min(_pairwise_dist2(X, C), axis=1)))
        if inertia < best[2]:
            best = (labels, C, inertia)
    return best


# ================================================================ 4. 手写多项 Logistic 回归
class SoftmaxRegression:
    """
    多项 Logistic 回归 (Softmax 回归), K 分类, 全批量梯度下降。

    模型:   z = X W + b            (n×d @ d×K -> n×K 个 logits)
            p_k = softmax(z)_k = exp(z_k) / Σ_j exp(z_j)
    损失:   加权交叉熵 + L2 正则
            L = -(1/n) Σ_i w_{y_i} · log p_{y_i}  +  λ||W||²
            (w 是类别权重, 用来对抗 喜欢:不喜欢 ≈ 1:60 的不平衡)
    梯度:   ∂L/∂W = (1/n) Xᵀ (p - y_onehot, 按样本权重加权) + 2λW
            ∂L/∂b = (1/n) Σ (p - y_onehot)
    更新:   W ← W - lr·∂L/∂W,  b ← b - lr·∂L/∂b

    数值稳定: softmax 前减去每行最大值 max(z),
    exp(z - max) / Σexp(z - max) 与原式数学等价但防止 exp 溢出。
    """

    def __init__(self, n_classes=2, lr=0.5, epochs=300, l2=1e-4,
                 class_weight=None):
        self.K, self.lr, self.epochs, self.l2 = n_classes, lr, epochs, l2
        self.class_weight = class_weight  # 形如 {0: w0, 1: w1}

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)     # 数值稳定
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        self.W = np.zeros((d, self.K))
        self.b = np.zeros(self.K)
        y1h = np.zeros((n, self.K))
        y1h[np.arange(n), y] = 1.0               # one-hot 编码
        # 每个样本的权重 = 其类别的权重
        sw = np.ones(n)
        if self.class_weight:
            for cls, w in self.class_weight.items():
                sw[y == cls] = w
        for ep in range(self.epochs):
            p = self._softmax(X @ self.W + self.b)         # 前向: 预测概率
            diff = (p - y1h) * sw[:, None]                 # 加权残差
            gW = X.T @ diff / n + 2 * self.l2 * self.W     # ∂L/∂W
            gb = diff.mean(axis=0)                         # ∂L/∂b
            self.W -= self.lr * gW                         # 梯度下降更新
            self.b -= self.lr * gb
        return self

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ================================================================ 5. 聚类 + 流派倾向标签
def cluster_music(df, n_pca, k):
    """标准化 -> 手写 PCA(SVD) -> 手写 KMeans。"""
    mu = df[COLUMNS].mean().values
    sd = df[COLUMNS].std().replace(0, 1).values
    Xs = (df[COLUMNS].values - mu) / sd                    # 标准化
    Xp, V, vr = my_pca(Xs, n_pca)
    print(f"[PCA-SVD] {Xs.shape[1]} -> {n_pca} 维, "
          f"累计解释方差 {vr[:n_pca].sum():.1%}")
    labels, C, inertia = my_kmeans(Xp, k)
    df["cluster"] = labels
    print(f"[KMeans] k={k}, 惯性 {inertia:.0f}, 各类歌曲数:")
    print(df["cluster"].value_counts().sort_index().to_string())
    return df, Xp


def genre_label(z, yr_mean):
    """把类别特征画像翻译成现实音乐特征/流派倾向标签 (MSD 无流派真值, 属推断)。"""
    tags = []
    if z.get("timbre_avg_2", 0) > 0.5:
        tags.append("明亮")
    elif z.get("timbre_avg_2", 0) < -0.5:
        tags.append("暗沉柔和")
    if z.get("timbre_avg_3", 0) > 0.5:
        tags.append("平滑抒情")
    elif z.get("timbre_avg_3", 0) < -0.5:
        tags.append("打击感强")
    if z.get("timbre_avg_1", 0) > 0.5:
        tags.append("高响度")
    elif z.get("timbre_avg_1", 0) < -0.5:
        tags.append("低响度")
    cov_z = z[[c for c in z.index if c.startswith("timbre_cov")]].mean()
    if cov_z > 0.4:
        tags.append("动态起伏大")
    if yr_mean < 1975:
        tags.append("经典老歌")
    elif yr_mean > 2000:
        tags.append("新世纪")
    return "·".join(tags) + "型" if tags else "均衡流行型"


def describe_clusters(df):
    print(f"\n{'=' * 70}\n聚类类别画像与流派倾向\n{'=' * 70}")
    mean = df[COLUMNS].mean()
    std = df[COLUMNS].std().replace(0, 1)
    labels = {}
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        z = ((sub[COLUMNS].mean() - mean) / std).drop("year")
        top = z.abs().sort_values(ascending=False).head(3)
        feats = ", ".join(f"{TIMBRE_MEANING.get(n, n)}({'高' if z[n] > 0 else '低'}"
                          f"{abs(z[n]):.1f}σ)" for n in top.index)
        yr = sub["year"][sub["year"] > 1900]
        labels[c] = genre_label(z, yr.mean())
        print(f"类别 {c:>2} | {len(sub):>6} 首 | 年代 {yr.mean():.0f} | "
              f"{labels[c]}\n        显著特征: {feats}")
    return labels


# ================================================================ 6. 300 用户评估
def evaluate_users(df, Xp, n_users, n_liked, neg_per_pos, rng):
    """
    核心: 对 n_users 个随机用户逐一建模 (无种子, 每次运行结果不同)。

    每个用户:
      a) 歌单: 80% 取自某个足够大的聚类 + 20% 随机;
      b) 歌单 8:2 划分训练/测试;
      c) 训练该用户的多项 Logistic 回归 (此处 K=2: 喜欢/不喜欢,
         softmax 退化为二元 logistic), 类别权重 = 反频率, 对抗不平衡;
      d) 召回率 = 测试喜欢中 predict=喜欢 的比例;
      e) 歌单平均推荐概率 = predict_proba(喜欢) 在歌单上的均值
         (softmax 输出是真实概率, 无需近似);
      f) 用户图上位置 = 歌单 PCA 坐标质心。
    """
    counts = df["cluster"].value_counts()
    big = counts[counts >= max(n_liked, 50)].index.to_numpy()
    n = len(df)
    recs, avg_probs, ux, uy, favs = [], [], [], [], []
    t0 = time.time()

    for uid in range(1, n_users + 1):
        fav = int(rng.choice(big))
        pool = df.index[df["cluster"] == fav].to_numpy()
        others = df.index[df["cluster"] != fav].to_numpy()
        n_main = min(int(n_liked * 0.8), len(pool))
        liked = np.concatenate([
            rng.choice(pool, size=n_main, replace=False),
            rng.choice(others, size=n_liked - n_main, replace=False)])
        pos = df.index.get_indexer(liked)
        rng.shuffle(pos)

        cut = max(int(len(pos) * 0.8), 1)              # ---- 8:2 划分 ----
        pos_tr, pos_te = pos[:cut], pos[cut:]
        unheard = np.setdiff1d(np.arange(n), pos)

        neg_tr = rng.choice(unheard,
                            size=min(len(pos_tr) * neg_per_pos, len(unheard)),
                            replace=False)
        X_tr = Xp[np.concatenate([pos_tr, neg_tr])]
        y_tr = np.concatenate([np.ones(len(pos_tr), int),
                               np.zeros(len(neg_tr), int)])
        # 类别权重 = 反频率: 少数类(喜欢)获得更大权重
        w = {0: len(y_tr) / (2 * (y_tr == 0).sum()),
             1: len(y_tr) / (2 * (y_tr == 1).sum())}
        clf = SoftmaxRegression(n_classes=2, lr=0.5, epochs=300,
                                l2=1e-4, class_weight=w)
        clf.fit(X_tr, y_tr)

        rec = float((clf.predict(Xp[pos_te]) == 1).mean()) if len(pos_te) else 0.0
        avg_p = float(clf.predict_proba(Xp[pos])[:, 1].mean())

        center = Xp[pos].mean(axis=0)
        recs.append(rec)
        avg_probs.append(avg_p)
        ux.append(center[0])
        uy.append(center[1])
        favs.append(fav)

        if uid % 100 == 0 or uid == n_users:
            print(f"[评估] {uid}/{n_users} 用户完成, 用时 {time.time() - t0:.0f}s, "
                  f"当前平均召回率 {np.mean(recs):.2%}")

    return pd.DataFrame({"user": range(1, n_users + 1), "fav_cluster": favs,
                         "recall": recs, "avg_like_prob": avg_probs,
                         "x": ux, "y": uy})


# ================================================================ 7. 可视化 (plt.show)
def show_cluster_users(Xp, df, labels, users, max_points=8000):
    """图1: 聚类染色散点 + 300 用户叠加 (星形, 颜色=召回率)。"""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(max_points, len(df)), replace=False)
    plt.figure(figsize=(12, 9))
    for c in sorted(df["cluster"].unique()):
        m = idx[df["cluster"].values[idx] == c]
        plt.scatter(Xp[m, 0], Xp[m, 1], s=4, alpha=0.35,
                    label=f"{c}: {labels[c]}")
    sc = plt.scatter(users["x"], users["y"], c=users["recall"],
                     cmap="RdYlGn", vmin=0, vmax=1,
                     s=60, edgecolors="black", linewidths=0.6,
                     marker="*", zorder=5)
    plt.colorbar(sc, label="用户召回率")
    plt.title("音乐聚类染色图 + 300 个随机用户位置 (星形, 颜色=召回率)")
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")
    plt.legend(markerscale=3, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.show()


def show_user_distributions(users):
    """图2: 召回率与歌单平均推荐概率分布直方图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(users["recall"], bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title(f"{len(users)} 个用户的测试集召回率分布")
    axes[0].set_xlabel("召回率")
    axes[0].set_ylabel("用户数")
    axes[1].hist(users["avg_like_prob"], bins=20, color="darkorange",
                 edgecolor="white")
    axes[1].set_title("歌单平均推荐概率分布")
    axes[1].set_xlabel("平均推荐概率")
    plt.tight_layout()
    plt.show()


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="音乐推荐 v5: 手写 numpy 核心算法")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--max-rows", type=int, default=150000)
    ap.add_argument("--clusters", type=int, default=12)
    ap.add_argument("--pca-dims", type=int, default=15)
    ap.add_argument("--users", type=int, default=300, help="随机用户数 (无种子)")
    ap.add_argument("--n-liked", type=int, default=30, help="每个用户歌单大小")
    ap.add_argument("--neg-per-pos", type=int, default=60,
                    help="训练负样本 = 训练喜欢数 × 该倍数")
    args = ap.parse_args()

    df = load_data(args.data, args.max_rows)
    df, Xp = cluster_music(df, args.pca_dims, args.clusters)
    labels = describe_clusters(df)

    rng = np.random.default_rng()  # 不设种子: 每次运行抽样不同的用户
    users = evaluate_users(df, Xp, args.users, args.n_liked,
                           args.neg_per_pos, rng)

    print(f"\n{'=' * 70}\n{args.users} 个随机用户总体结果\n{'=' * 70}")
    print(f"召回率:          平均 {users['recall'].mean():.2%} | "
          f"中位数 {users['recall'].median():.2%} | "
          f"召回率>50%的用户占比 {(users['recall'] > 0.5).mean():.1%}")
    print(f"歌单平均推荐概率: 平均 {users['avg_like_prob'].mean():.2%} | "
          f"中位数 {users['avg_like_prob'].median():.2%}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "users_result.csv")
    users.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[保存] 每个用户的明细: {out}")

    show_cluster_users(Xp, df, labels, users)   # 图1: plt.show()
    show_user_distributions(users)              # 图2: plt.show()


if __name__ == "__main__":
    main()
