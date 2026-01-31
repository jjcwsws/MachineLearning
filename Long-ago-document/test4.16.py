import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


# 加载数据集
data = pd.read_csv('heart.csv')

# 显示前5行
print(data.head())


# 分割特征和标签
X = data.drop('target', axis=1)  # 特征
y = data['target']               # 标签

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 创建决策树分类器
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 绘制决策树
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'])
plt.show()
