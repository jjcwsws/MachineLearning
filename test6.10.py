# 导入所需库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# ----------------------
# 1. 数据加载与预处理
# ----------------------
# 加载数据集（假设heart.csv与代码同目录）
data = pd.read_csv('heart.csv')
X = data.drop('target', axis=1) # 特征
y = data['target'] # 标签（0=无心脏病，1=有心脏病）

# 划分训练集与测试集（7:3，保持标签分布一致）
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42, stratify=y
)

# ----------------------
# 2. 随机森林模型构建与调参
# ----------------------
# 初始化模型
rf = RandomForestClassifier(random_state=42) # random_state确保结果可复现

# 定义超参数搜索空间
param_grid = {
'n_estimators': [100, 200, 300], # 树的数量
'max_depth': [None, 10, 20, 30], # 树的最大深度（None表示不限制）
'min_samples_split': [2, 5, 10], # 节点分裂所需最小样本数
'max_features': ['auto', 'sqrt'] # 分裂时考虑的最大特征数
}

# 网格搜索（5折交叉验证，以ROC-AUC为优化目标）
grid_search = GridSearchCV(
estimator=rf,
param_grid=param_grid,
cv=5,
scoring='roc_auc',
n_jobs=-1 # 使用全部CPU核心加速
)
grid_search.fit(X_train, y_train)

# 获取最优模型和参数
best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_
print("最优超参数:", best_params)

# ----------------------
# 3. 模型评估
# ----------------------
# 预测测试集
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1] # 正类概率

# 计算评估指标
print("\n=== 评估结果 ===")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("分类报告:\n", classification_report(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))

# ----------------------
# 4. 特征重要性可视化
# ----------------------
features = X.columns
importances = best_rf.feature_importances_

# 按重要性排序
sorted_indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_indices)), importances[sorted_indices], align='center')
plt.xticks(range(len(sorted_indices)), features[sorted_indices], rotation=45, ha='right')
plt.xlabel('特征')
plt.ylabel('重要性得分')
plt.title('随机森林特征重要性排名')
plt.show()

# ----------------------
# 5. 可选：处理数据不平衡（如正负样本比例失衡）
# ----------------------
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# best_rf.fit(X_train_resampled, y_train_resampled)
# # 重新评估...