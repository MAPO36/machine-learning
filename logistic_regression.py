import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
df = pd.read_csv('data.csv')
rows = df.shape[0]
cols = df.shape[1]
print(f'共有{rows-1}项数据参与测试, 共有 {cols-2}项特征指标')
print('数据样例')
print(df.head())
x_raw = df.iloc[:,2:].values
y_raw = df.iloc[:,1].values
y_raw[y_raw == 'M'] = 1
y_raw[y_raw == 'B'] = 0
y_raw = y_raw.astype(int)
y_raw = y_raw.reshape(-1,1)
x_orig = x_raw.copy()
poly = PolynomialFeatures(degree = 2, include_bias = False)
x_raw = poly.fit_transform(x_raw)
x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size = 0.2, random_state = 42, stratify = y_raw)
#缩放，为了防止信息泄露，我们只能用训练集的mean和std来缩放测试集
mean = np.mean(x_train, axis = 0)
std = np.std(x_train, axis = 0)
x_train = (x_train - mean) / std#455*459
x_test = (x_test - mean) / std
#偏置
ones = np.ones((x_train.shape[0], 1))
x_train = np.concatenate((x_train, ones), axis = 1)
ones = np.ones((x_test.shape[0], 1))
x_test = np.concatenate((x_test, ones), axis = 1)
def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))
epoch = 500
alpha = 0.01
lambda_ = 0.001
w = np.full((x_train.shape[1], 1), 0.1)
cost_list = []
for i in range(epoch):
    w_reg = w.copy()
    w_reg[-1] = 0
    #log 0 会引发数学错误，所以需要eps以避免
    eps = 1e-15
    #cost需要正则化
    y_hat = sigmoid(x_train @ w)
    cost = -1 * (np.sum(y_train * np.log(y_hat + eps) + (1 - y_train) * np.log(1 - y_hat + eps)) + lambda_ * np.sum(w_reg ** 2)) / (x_train.shape[0])
    cost_list.append(cost)
    grad = (x_train.T @ (y_hat - y_train) + lambda_ * w_reg) / x_train.shape[0]
    w = w - alpha * grad
y_hat_test = sigmoid(x_test @ w)
y_hat_class = (y_hat_test - 0.5 > 0).astype(int)
accuracy = np.mean((y_hat_class == y_test).astype(int))
print(f"测试集合上预测精准度：{accuracy}")
def plot_cost(epoch, cost_list):
    """
    画出损失函数随训练次数的图
    :param epoch: 训练次数
    :param cost_list: 损失函数随训练数从零开始的列表
    """
    plt.figure(figsize = (12,6))
    x1 = range(epoch)
    y1 = cost_list
    plt.plot(x1, y1, color='blue', label='cost curve')
    plt.title('cost function')
    plt.xlabel('training number')
    plt.ylabel('cost')
    plt.legend()
    plt.show()


from sklearn.linear_model import LogisticRegression
def plot_pca_decision_boundary(X_orig, y_orig):
    """
    专门用来可视化PCA降维后的数据分布和决策边界的函数
    X_orig: 传入最原始的数据 (即 df.iloc[:,2:].values)，不要多项式展开的
    y_orig: 标签 y_raw
    """
    # 1. 预处理：标准化原始数据 (PCA 对数据缩放非常敏感)
    X_scaled = (X_orig - np.mean(X_orig, axis=0)) / np.std(X_orig, axis=0)

    # 2. PCA 降维到 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 3. 训练一个用于可视化的简单逻辑回归模型
    # 我们只用 PCA 后的 2 个维度来训练，目的是画出清晰的决策边界
    clf = LogisticRegression()
    clf.fit(X_pca, y_orig.ravel())

    # 4. 准备绘图网格
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    # 5. 预测网格区域
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 6. 绘图
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    # 绘制散点
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_orig.flatten(), cmap='coolwarm', edgecolor='k', s=40)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA 2D Decision Boundary (Data Visualization)')
    plt.show()
# 注意一定要传 x_orig，不要传 x_train，因为 PCA 不需要多项式展开
plot_pca_decision_boundary(x_orig, y_raw)
plot_cost(epoch, cost_list)
