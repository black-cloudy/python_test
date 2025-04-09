
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定义Sigmoid函数
def sigmoid(x, a, b, c):
    # 对值进行截断以避免溢出
    z = np.clip(a * (x - b), -500, 500)
    return c / (1 + np.exp(-z))

# 定义R^2计算函数
def calculate_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# 读取CSV文件
file_path = './linear_current_force_2.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 提取电流和肌电信号列
current = data.iloc[:, 0].values
emg_signal = data.iloc[:, 1].values

# 设置迭代参数
desired_r_squared = 0.99
max_iterations = 10000
tolerance = 1e-6

# 初始猜测参数 [a, b, c]
initial_guess = [1, 0, max(emg_signal)]
best_params = initial_guess
best_r_squared = 0

# 迭代优化
for iteration in range(max_iterations):
    bounds = ([0, -np.inf, 0], [np.inf, np.inf, np.inf])  # 确保a为正数，c为正数
    popt, pcov = curve_fit(sigmoid, current, emg_signal, p0=initial_guess, bounds=bounds, maxfev=500000)

    emg_fit = sigmoid(current, *popt)
    r_squared = calculate_r_squared(emg_signal, emg_fit)

    if r_squared > best_r_squared:
        best_r_squared = r_squared
        best_params = popt

    if r_squared >= desired_r_squared:
        break

    # 随机调整初始猜测参数以进行下一次迭代
    initial_guess = popt + np.random.normal(0, tolerance, size=len(popt))

# 输出拟合结果
print(f"Best parameters: a = {best_params[0]}, b = {best_params[1]}, c = {best_params[2]}")
print(f"Best R^2: {best_r_squared}")

# 计算最终拟合曲线
current_fit = np.linspace(min(current), max(current), 1000)
emg_fit_final = sigmoid(current_fit, *best_params)


# 重新计算采样点
sampled_indices = np.linspace(0, len(current_fit) - 1, 803, dtype=int)       # 803重采样数据点数量
sampled_current = current_fit[sampled_indices]
sampled_emg = emg_fit_final[sampled_indices]

# 创建一个新的DataFrame
output_data = pd.DataFrame({
    'Current': sampled_current,
    'EMG_Signal': sampled_emg
})

# 保存到CSV文件
output_file_path = 'E:/FJY/2024_8_17/1/sin/fitted_data_2.csv'  # 更改文件名
output_data.to_csv(output_file_path, index=False)
print(f"拟合数据已保存到 {output_file_path}")

# 绘制原始数据和拟合结果
plt.figure()
plt.scatter(current, emg_signal, label='Original Data', color='b', s=10)
plt.plot(current_fit, emg_fit_final, label='Sigmoid Fit', color='r')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('EMG Signal vs Current with Sigmoid Fit')
plt.xlabel('Current')
plt.ylabel('EMG Signal')

# 显示图形
plt.show()



