import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 读取数据
data = pd.read_csv('heart.csv')
X = data.drop('target', axis=1)
y = data['target']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义模型（使用默认参数）
models = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(random_state=42)
}

# 训练并评估模型
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        '准确率': accuracy_score(y_test, y_pred),
        '精确率': precision_score(y_test, y_pred),
        '召回率': recall_score(y_test, y_pred),
        'F1分数': f1_score(y_test, y_pred)
    }

# 输出结果表格
print('模型性能对比：')
print(pd.DataFrame(results).T.round(5))