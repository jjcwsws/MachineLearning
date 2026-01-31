# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import LinearRegression
#
# # 训练数据
# diameter = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)  # 直径（英寸）
# price = np.array([7, 9, 13, 17.5, 18])  # 价格（美元）
#
# # 绘制散点图
# plt.scatter(diameter, price, color='blue', label='数据点')
# plt.title('披萨价格与直径的关系')
# plt.xlabel('披萨直径（英寸）')  # X轴标注
# plt.ylabel('披萨价格（美元）')  # Y轴标注
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # 创建线性回归模型
# model = LinearRegression()
# model.fit(diameter, price)
#
# # 拟合直线方程
# slope = model.coef_[0]
# intercept = model.intercept_
# print(f'拟合直线方程: y = {slope:.2f} * x + {intercept:.2f}')
#
# # 预测12英寸披萨的价格
# predicted_price = model.predict(np.array([[12]]))
# print(f'预测12英寸披萨的价格: {predicted_price[0]:.2f}美元')
#
# # 评价模型的准确率
# r_squared = model.score(diameter, price)
# print(f'模型的R^2值: {r_squared:.4f}')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 使负号正常显示
# 训练样本
# 直径（英寸）
X = np.array([[6], [8], [10], [14], [18]])
# 价格（美元）
y = np.array([7, 9, 13, 17.5, 18])
# 创建线性回归模型
model = LinearRegression()
# 拟合模型
model.fit(X, y)
# 预测12英寸披萨的价格
predicted_price = model.predict(np.array([[12]]))
# 输出预测结果
print(f"预测12英寸披萨的价格: ${predicted_price[0]:.3f}")
# 评价模型的准确率
r_squared = model.score(X,y)
print(f'模型的R^2值: {r_squared:.4f}')
# 画散点图和拟合直线
plt.scatter(X, y, color='blue', label='实际数据')
plt.plot(X, model.predict(X), color='red', label='拟合直线')
plt.scatter([12], predicted_price, color='green', label='预测价格', zorder=5)
plt.title('披萨价格与直径的关系')
plt.xlabel('直径 ')
plt.ylabel('价格 ')
plt.legend()
plt.show()

# # 评价模型的准确率
# r_squared = model.score(X,y)
# print(f'模型的R^2值: {r_squared:.4f}')



# import matplotlib.pyplot as plt
#
# # 创建散点数据
# x = [6, 8, 10, 14, 18]
# y = [7, 9, 13, 17.5, 18]
#
# # 绘制散点图
# plt.scatter(x, y, color='blue', label='实际数据')
#
# # 设置背景颜色
# plt.gcf().set_facecolor('white')
#
# # 其他绘图代码...
# plt.title('披萨价格与直径的关系')
# plt.xlabel('直径 (英寸)')
# plt.ylabel('价格 ($)')
# plt.legend()
# plt.show()


#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 设置字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
# plt.rcParams['axes.unicode_minus'] = False    # 使负号正常显示
#
# x = [6, 8, 10, 14, 18]
# y = [7, 9, 13, 17.5, 18]
#
# plt.scatter(x, y, color='blue', label='实际数据')
#
# plt.title('披萨价格与直径的关系')
# plt.xlabel('直径 (英寸)')
# plt.ylabel('价格 ($)')
# plt.legend()
#
# plt.show()
#
# from sklearn.linear_model import LinearRegression
# import numpy as np
#
# X = np.array([[6], [8], [10], [14], [18]])
# y= np.array([7, 9, 13, 17.5, 18])
#
# model = LinearRegression()
# model.fit(X,y)
#
# slope = model.coef_[0]
# intercept = model.intercept_
# print(f'拟合直线方程: y = {slope:.2f} * x + {intercept:.2f}')
#
# predicted_price = model.predict(np.array([[12]]))
# print(f'预测12英寸披萨的价格: {predicted_price[0]:.3f}美元')
#
# r_squared = model.score('直径 (英寸)', '价格 ($)')
# print(f'模型的R^2值: {r_squared:.4f}')

