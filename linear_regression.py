import numpy as np
import matplotlib.pyplot as plt
#准备数据集
np.random.seed(42)
m = 200
x_raw = np.random.uniform(low=1, high=100, size=(m, 2))
#真实的方程
y_raw = 3*x_raw[:, 0] + 3*(x_raw[:, 1]**2) + 5 + np.random.randn(m)
y_raw =  y_raw.reshape(-1, 1)
#正则化
x1_mean = np.mean(x_raw[:, 0])
x2_mean = np.mean(x_raw[:, 1])
sigma1 = np.std(x_raw[:, 0])
sigma2 = np.std(x_raw[:, 1])
x1 = (x_raw[:, 0] - x1_mean) / sigma1
x2 = (x_raw[:, 1] - x2_mean) / sigma2
#特征整理为[x1,x1^2,x2,x2^2,x1*x2]形式
x1_2 = x1 ** 2
x2_2 = x2 ** 2
x12 = x1 * x2
X = np.column_stack([x1, x1_2, x2, x2_2, x12])
ones = np.ones((m, 1))
X = np.hstack([X, ones])
#设定初始参数
w = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 5]).reshape(-1, 1)
alpha = 0.01
lambda_ = 0.1
epoch = 500
J_list = []
for i in range(epoch):
    error = (X @ w) - y_raw
    w_reg = w.copy()
    w_reg[5] = 0
    J_i = (np.sum(error ** 2) + lambda_ * np.sum(w_reg ** 2)) / (2*m)
    J_list.append(J_i)
    J_grad = (X.T @ error + lambda_ * w_reg) / m#5x1
    w = w - alpha * J_grad
print("final W = ", w)
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(range(epoch), J_list, color='blue')
ax1.set_xlabel('training number')
ax1.set_ylabel('cost')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
x11, x22 = np.meshgrid(np.linspace(-2, 2,50), np.linspace(-2, 2,50))
y_hat = w[0] * x11 + w[1] * (x11 ** 2) + w[2] * x22 + w[3] * (x22 ** 2) + w[4] * (x11 * x22) + w[5]
ax2.scatter(x1.flatten(), x2.flatten(), y_raw.flatten(), color='red', alpha=0.5, label='real data')
ax2.plot_surface(x11, x22, y_hat, color='blue', alpha=0.5, label='predicted data')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
ax2.set_title("compare tow surfaces")
ax2.legend()
plt.tight_layout()
plt.show()