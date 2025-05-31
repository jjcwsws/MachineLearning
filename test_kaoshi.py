import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签数据
feature_names = iris.feature_names  # 特征名称
target_names = iris.target_names  # 类别名称

# 转换为DataFrame并显示前5行
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['target'] = y
iris_df['species'] = [target_names[i] for i in y]
print(iris_df.head())

# 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化（KNN对尺度敏感）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 创建KNN分类器（K=3）
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 预测并计算准确率
train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)

print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")


# 选择前两个特征（花萼长度和宽度）进行可视化
X_vis = X[:, :2]
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.3, random_state=42)

# 重新训练模型（仅用两个特征）
knn_vis = KNeighborsClassifier(n_neighbors=3)
knn_vis.fit(X_train_vis, y_train_vis)

# 创建网格点
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点类别
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')

# 绘制训练样本点
scatter = plt.scatter(X_train_vis[:, 0], X_train_vis[:, 1], c=y_train_vis,
                      cmap='viridis', edgecolor='k', s=50, label='Train')
plt.scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test_vis,
            cmap='viridis', marker='x', s=100, label='Test')

# 添加图例和标签
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("KNN Classification (k=3) on Iris Dataset")
plt.legend()
plt.colorbar(scatter, ticks=[0, 1, 2], label='Species')
plt.show()