import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from computeCost import *
from gradientDescent import *
from plotData import *

# ===================== Part 1: Plotting =====================
print('Plotting Data...')
data = np.loadtxt('ex1data1.txt', delimiter=',', usecols=(0, 1))
X = data[:, 0]  # X:代表特征/输入变量
y = data[:, 1]  # y:代表目标变量/输出变量
m = y.size      # 代表训练集中实例的数量

plt.ion()       # 打开交互模式
plt.figure(1)
plot_data(X, y)

input('Program paused 1. Press ENTER to continue')

# ===================== Part 2: Gradient descent 梯度下降法 =====================
print('Running Gradient Descent...')

X = np.c_[np.ones(m), X]    # Add a column of ones to X
theta = np.zeros(2)         # initialize fitting parameters.  初始化拟合参数

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print("Testing the cost function ...")
# Compute and display initial cost
J = compute_cost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = %f' %(J))
print('Expected cost value (approx) 32.07')

# further testing of the cost function
J = compute_cost(X, y, [-1, 2])
print('With theta = [-1, 2]\nCost computed = %f' %(J))
print('Expected cost value (approx) 54.24')
input('Program paused 2. Press enter to continue.')


print('Running Gradient Descent ...')
# run gradient descent
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: ' + str(theta.reshape(2)))
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
plt.figure(1)   # 保持之前的情节可见，在figure 1 基础上绘制
line1, = plt.plot(X[:, 1], np.dot(X, theta), label='Linear Regression')
# plt.legend(handles=[line1])
plt.legend(loc='upper right')

input('Program paused 3. Press ENTER to continue')

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict2*10000))

input('Program paused 4. Press ENTER to continue')

# ===================== Part 3: Visualizing J(theta0, theta1) =====================
print('Visualizing J(theta0, theta1) ...')

# 依照定义间隔生成均匀分布的数值
theta0_vals = np.linspace(start=-10, stop=10, num=100)
theta1_vals = np.linspace(start=-1, stop=4, num=100)

xs, ys = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros(xs.shape)

# Fill out J_vals
for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = compute_cost(X, y, t)

J_vals = np.transpose(J_vals)

fig1 = plt.figure(2)
ax = fig1.gca(projection='3d')
ax.plot_surface(xs, ys, J_vals)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')

plt.figure(3)
# logspace 创建等比数列
lvls = np.logspace(start=-2, stop=3, num=20)
plt.contour(xs, ys, J_vals, levels=lvls, norm=LogNorm())
plt.plot(theta[0], theta[1], c='r', marker="x")

input('ex1 Finished. Press ENTER to exit')
