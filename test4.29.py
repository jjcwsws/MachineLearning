# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn import datasets
#
# X,y = datasets.make_moons(noise = 0.15,random_state = 666)
#
# plt.scatter(X[y == 0,0],X[y == 0,1])
# plt.scatter(X[y == 1,0],X[y == 1,1])
# plt.show()


# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
#
# X,y = datasets.make_moons(noise = 0.15,random_state = 666)
#
# def RBFKernelSVC(gamma = 1.0):
#     return Pipeline([
#         ('std_scaler',StandardScaler()),
#         ('svc', SVC(kernel ='rbf',gamma = gamma))
#     ])
#
#
#
# # svc_gamma01 = RBFKernelSVC(gamma = 1.0)
# # svc.fit(X,y)
#
# # svc_gamma01 = RBFKernelSVC(gamma = 0.1)
# # svc_gamma01.fit(X,y)
#
# svc_gamma05 = RBFKernelSVC(gamma = 0.5)
# svc_gamma05.fit(X,y)
#
#
#
# def plot_decision_boundary(model, axis):
#     x0,x1 = np.meshgrid(
#         np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
#         np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
#     )
#
#     X_new = np.c_[x0.ravel(), x1.ravel()]
#     y_predict = model. predict(X_new)
#     zz =y_predict.reshape(x0. shape)
#
#     from matplotlib.colors import ListedColormap
#     custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D','#90CAF9'])
#
#     plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
#
# plot_decision_boundary(svc_gamma05, axis=[-1.5, 2.5, -1.0, 1.5])
# plt.scatter(X[y == 0, 0], X[y == 0, 1])
# plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show()





# # 导入所需库
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
#
# # 加载鸢尾花数据集
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# # 划分训练集和测试集（70:30）
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )
#
# # 创建线性核SVM分类器并训练
# clf = SVC(kernel="linear")
# clf.fit(X_train, y_train)
#
# # 预测并输出准确率
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("分类任务测试集准确率：{:.2f}%".format(accuracy * 100))




# 导入所需库
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 加载糖尿病数据集
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 划分训练集和测试集（70:30）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 特征标准化（SVM对数据尺度敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建RBF核SVM回归器并训练
regressor = SVR(kernel="rbf")
regressor.fit(X_train_scaled, y_train)

# 预测并输出评估指标
y_pred = regressor.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("回归任务均方误差（MSE）：{:.2f}".format(mse))
print("回归任务R²分数：{:.2f}".format(r2))