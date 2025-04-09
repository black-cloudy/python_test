#
# #--------------------------------------------------------------------------------生成伪随机波形
# import numpy as np
# import matplotlib.pyplot as plt
#
# def generate_irregular_waveform(num_points, min_val, max_val, noise_level, smoothing_window=50, seed=42):
#     """
#     生成不规则的平滑波形，带有噪声并且能够设定最大值和最小值
#
#     :param num_points: 波形的点数
#     :param min_val: 波形的最小值
#     :param max_val: 波形的最大值
#     :param noise_level: 噪声强度
#     :param smoothing_window: 平滑窗口大小
#     :param seed: 随机数种子，用于生成相同的伪随机波形
#     :return: 不规则的平滑波形
#     """
#     np.random.seed(seed)  # 保证每次生成的波形相同
#
#     # 混合不同频率和振幅的正弦波
#     base_waveform = (np.sin(np.linspace(0, 4 * np.pi, num_points)) +
#                      0.5 * np.sin(np.linspace(0, 2 * np.pi, num_points)) +
#                      0.2 * np.sin(np.linspace(0, 13 * np.pi, num_points)))
#
#     # 添加随机噪声
#     noise = np.random.normal(0, noise_level, num_points)
#
#     # 将噪声加入基础波形
#     random_waveform = base_waveform + noise
#
#     # 使用卷积对波形进行平滑处理
#     smoothed_waveform = np.convolve(random_waveform, np.ones(smoothing_window) / smoothing_window, mode='same')
#
#     # 将平滑后的波形归一化到指定的最小值和最大值之间
#     smoothed_waveform = (smoothed_waveform - np.min(smoothed_waveform)) / (np.max(smoothed_waveform) - np.min(smoothed_waveform))
#     smoothed_waveform = smoothed_waveform * (max_val - min_val) + min_val
#
#     return smoothed_waveform
#
# # 参数设置
# num_points = 500        # 数据点的数量
# min_val = 7             # 波形的最小值
# max_val = 30            # 波形的最大值
# noise_level = 0.1       # 噪声强度
# smoothing_window = 60   # 平滑窗口大小
# seed = 42               # 随机数种子
#
# # 生成不规则的平滑波形
# irregular_waveform = generate_irregular_waveform(num_points, min_val, max_val, noise_level, smoothing_window, seed)
#
# # 绘制波形
# t = np.arange(num_points)
# plt.plot(t, irregular_waveform, label='Generated irregular waveform')
#
# # 设置标签和标题
# plt.title('Irregular Smooth Waveform')
# plt.xlabel('Time point (k)')
# plt.ylabel('Dorsiflexion angle (°)')
# plt.grid(True)
# plt.legend()
#
# # 显示图形
# plt.show()

# -----------------------------------------------------------滑窗无平滑处理肌电数据
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import interpolate
#
# # Read the .csv file
# df = pd.read_csv('sin_current_force_1.csv')
#
# # Select the fourth column (index starts at 0, so column 4 is at index 3)
# fourth_column = df.iloc[:, 3]
#
# # Group every 10 data points and calculate the mean for each group
# averages = fourth_column.groupby(fourth_column.index // 25).mean()
#
# # Plot the original averaged data
# plt.plot(averages, label='Averaged Data')
# plt.title('Averaged Data')
# plt.xlabel('Group Index')
# plt.ylabel('Average Value')
# plt.legend()
# plt.show()
#
# # Resample the data to 1000 points using interpolation
# x_original = np.linspace(0, len(averages) - 1, len(averages))
# x_new = np.linspace(0, len(averages) - 1, 1000)
# f_interpolate = interpolate.interp1d(x_original, averages, kind='linear')
# resampled_data = f_interpolate(x_new)
#
# # Plot the resampled data
# plt.plot(x_new, resampled_data, label='Resampled Data (1000 points)', color='orange')
# plt.title('Resampled Data')
# plt.xlabel('Resampled Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
#
# # Optionally, save the resampled data to a file
# resampled_df = pd.DataFrame(resampled_data, columns=['Resampled Data'])
# resampled_df.to_csv('resampled_data.csv', index=False)
#
# print("Resampled data saved to 'resampled_data.csv'")
#

#---------------------------------------------------------------------------------------滑窗平滑处理肌电数据
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import interpolate
# from scipy.signal import savgol_filter
#
# # 读取 .csv 文件
# df = pd.read_csv('sin_current_force_2.csv')
#
# # 选择第四列（从0开始计数，所以第四列是索引3）
# fourth_column = df.iloc[:, 3]
#
# # 每10个数据取平均值
# averages = fourth_column.groupby(fourth_column.index // 25).mean()
#
# # 使用 Savitzky-Golay 滤波器平滑处理
# smoothed_averages = savgol_filter(averages, window_length=11, polyorder=3)
#
# # 删除 NaN 值，并同步调整 x_original
# valid_indices = ~np.isnan(smoothed_averages)
# x_original = np.linspace(0, len(smoothed_averages) - 1, len(smoothed_averages))[valid_indices]
# smoothed_averages_clean = smoothed_averages[valid_indices]
#
# # 使用线性插值将数据重采样到1000个点
# x_new = np.linspace(0, len(smoothed_averages_clean) - 1, 1000)
# f_interpolate = interpolate.interp1d(x_original, smoothed_averages_clean, kind='linear')
# resampled_data = f_interpolate(x_new)
#
# # 绘制重采样后的数据
# plt.plot(x_new, resampled_data, color='gray', alpha=0.5, label='Resampled Smoothed Data (1000 points)')
# plt.title('Resampled Smoothed Data')
# plt.xlabel('Resampled Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
#
# # 保存重采样数据到文件
# resampled_df = pd.DataFrame(resampled_data, columns=['Resampled Smoothed Data'])
# resampled_df.to_csv('resampled_smoothed_data.csv', index=False)
#
# print("重采样平滑数据已保存到 'resampled_smoothed_data.csv'")

# # # ------------------------------------------------------------------------------------------画出阴影偏差--sin_r
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
def sine_wave(min_value, max_value, frequency, duration, sampling_rate, shift_fraction=0.0):
    amplitude = (max_value - min_value) / 2.0
    offset = (max_value + min_value) / 2.0
    expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    expect_force_waveform = amplitude * np.sin(
        2 * np.pi * frequency * (expect_force_t + shift_fraction * duration) - np.pi / 2) + offset
    return expect_force_waveform


def triangle_wave(min_value, max_value, period, duration, sampling_rate, shift_fraction=1.0):
    amplitude = (max_value - min_value) / 2.0
    offset = (max_value + min_value) / 2.0
    expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # 使用 scipy.signal.sawtooth 来生成三角波，triangle 参数设置为 0.5 生成对称三角波
    expect_force_waveform = amplitude * signal.sawtooth(2 * np.pi * (expect_force_t + shift_fraction * period) / period,0.5) + offset
    return expect_force_waveform
def calculate_r2(y_true, y_pred):
    """
    计算偏差R²值。

    参数:
    y_true -- 实际值数组
    y_pred -- 预测值数组

    返回:
    R²值
    """
    # 计算均方误差
    ss_res = np.sum((y_true - y_pred) ** 2)

    # 计算总方差
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # 计算R²
    r2 = 1 - (ss_res / ss_tot)

    return r2
expect_force=sine_wave(1,6,0.2,10,100)
# expect_force=sine_wave(0,11,0.2,10,100)
# 读取CSV文件，假设文件名为'data.csv'
data = pd.read_csv('sine_ring_varify_1_alldata.csv')
# expect_force=data.iloc[:, 9]
# expect_force=[x / 6.5 for x in expect_force]
# 选择前7列作为原始数据，假设它们位于第1到第7列
raw_data = data.iloc[:, 0:3]
# raw_data = [x / 10 for x in raw_data]
raw_data = raw_data / 1

# 选择第8列作为均值数据，假设均值数据位于第8列
mean = data.iloc[:, 3]
mean = [x / 1 for x in mean]
# mean1=mean
# 计算7列原始数据的标准差，作为偏差范围
std_dev = raw_data.std(axis=1)
rmse = np.sqrt(np.mean((mean - expect_force) ** 2))
r2_value = calculate_r2(mean, expect_force)
print(f"R²值: {r2_value}")
# 如果第一列是时间列，可以用作时间轴 t
# 否则用索引作为时间轴
# if 'time' in data.columns:
#     t = data['time']  # 假设时间列名为'time'
# else:
#     t = np.arange(len(data))  # 使用索引作为时间
t = np.linspace(0, 10, len(data))  # 生成从0到10的等间距时间轴
# print(len(t))
# print(len(mean))
# print(len(expect_force))
# 绘制均值线
plt.figure(figsize=(8, 6))
plt.plot(t,mean, color='darkorange',alpha=1,linewidth=2, label='Mean')
plt.plot(t,expect_force,color='black',alpha=0.8, linewidth=2,label='Expected Force',linestyle='--')
# 绘制阴影表示偏差，±1标准差
plt.fill_between(t,mean - std_dev, mean + std_dev, color='peachpuff', alpha=0.5)
plt.text(0.5, 0.7, f'RMSE: {rmse:.2f}', horizontalalignment='center',
         verticalalignment='center', transform=plt.gca().transAxes, fontsize=18)
# 图像配置
plt.title('Sine/Ring', fontsize=18)
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Force (N)', fontsize=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right',fontsize=15)
# plt.grid(True)
plt.show()
# # ------------------------------------------------------------------------------------------画出阴影偏差--sanjiao_r
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# def sine_wave(min_value, max_value, frequency, duration, sampling_rate, shift_fraction=0.0):
#     amplitude = (max_value - min_value) / 2.0
#     offset = (max_value + min_value) / 2.0
#     expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#     expect_force_waveform = amplitude * np.sin(
#         2 * np.pi * frequency * (expect_force_t + shift_fraction * duration) - np.pi / 2) + offset
#     return expect_force_waveform
#
#
# def triangle_wave(min_value, max_value, period, duration, sampling_rate, shift_fraction=1.0):
#     amplitude = (max_value - min_value) / 2.0
#     offset = (max_value + min_value) / 2.0
#     expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#
#     # 使用 scipy.signal.sawtooth 来生成三角波，triangle 参数设置为 0.5 生成对称三角波
#     expect_force_waveform = amplitude * signal.sawtooth(2 * np.pi * (expect_force_t + shift_fraction * period) / period,0.5) + offset
#     return expect_force_waveform
# def calculate_r2(y_true, y_pred):
#     """
#     计算偏差R²值。
#
#     参数:
#     y_true -- 实际值数组
#     y_pred -- 预测值数组
#
#     返回:
#     R²值
#     """
#     # 计算均方误差
#     ss_res = np.sum((y_true - y_pred) ** 2)
#
#     # 计算总方差
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#
#     # 计算R²
#     r2 = 1 - (ss_res / ss_tot)
#
#     return r2
#
# expect_force=triangle_wave(0,5,5,10,100)
# # print(expect_force)
# # expect_force=sine_wave(0,11,0.2,10,100)
# # 读取CSV文件，假设文件名为'data.csv'
# data = pd.read_csv('triangle_ring_varify_1_alldata.csv')
# # expect_force=data.iloc[:, 9]
# # expect_force=[x / 6.5 for x in expect_force]
# # 选择前7列作为原始数据，假设它们位于第1到第7列
# raw_data = data.iloc[:, 0:3]
# # raw_data = [x / 10 for x in raw_data]
# raw_data = raw_data / 1
#
# # 选择第8列作为均值数据，假设均值数据位于第8列
# mean = data.iloc[:, 3]
# mean = [x / 1 for x in mean]
# # 计算7列原始数据的标准差，作为偏差范围
# std_dev = raw_data.std(axis=1)
# rmse = np.sqrt(np.mean((mean - expect_force) ** 2))
# r2_value = calculate_r2(mean, expect_force)
# print(f"R²值: {r2_value}")
# # 如果第一列是时间列，可以用作时间轴 t
# # 否则用索引作为时间轴
# # if 'time' in data.columns:
# #     t = data['time']  # 假设时间列名为'time'
# # else:
# t = np.linspace(0, 10, len(data))  # 生成从0到10的等间距时间轴
# print(len(t))
# print(len(mean))
# print(len(expect_force))
# # 绘制均值线
# plt.figure(figsize=(8, 6))
# plt.plot(t,mean, color='darkorange',alpha=1,linewidth=2, label='Mean')
# plt.plot(t,expect_force,color='black',alpha=0.8, linewidth=2,label='Expected Force',linestyle='--')
# plt.ylim([-1.5, 7])
# # 绘制阴影表示偏差，±1标准差
# plt.fill_between(t,mean - std_dev, mean + std_dev, color='peachpuff', alpha=0.5)
# plt.text(0.5, 0.7, f'RMSE: {rmse:.2f}', horizontalalignment='center',
#          verticalalignment='center', transform=plt.gca().transAxes, fontsize=18)
# # 图像配置
# plt.title('Triangle/Ring', fontsize=18)
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('Force (N)', fontsize=15)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(loc='upper right',fontsize=15)
# # plt.grid(True)
# plt.show()

#------------------------------------------------------------------------------------------画出阴影偏差--不规则波形期望力_r
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# def sine_wave(min_value, max_value, frequency, duration, sampling_rate, shift_fraction=0.0):
#     amplitude = (max_value - min_value) / 2.0
#     offset = (max_value + min_value) / 2.0
#     expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#     expect_force_waveform = amplitude * np.sin(
#         2 * np.pi * frequency * (expect_force_t + shift_fraction * duration) - np.pi / 2) + offset
#     return expect_force_waveform
#
#
# def triangle_wave(min_value, max_value, period, duration, sampling_rate, shift_fraction=1.0):
#     amplitude = (max_value - min_value) / 2.0
#     offset = (max_value + min_value) / 2.0
#     expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#
#     # 使用 scipy.signal.sawtooth 来生成三角波，triangle 参数设置为 0.5 生成对称三角波
#     expect_force_waveform = amplitude * signal.sawtooth(2 * np.pi * (expect_force_t + shift_fraction * period) / period,0.5) + offset
#     return expect_force_waveform
#
#
# def calculate_r2(y_true, y_pred):
#     """
#     计算偏差R²值。
#
#     参数:
#     y_true -- 实际值数组
#     y_pred -- 预测值数组
#
#     返回:
#     R²值
#     """
#     # 计算均方误差
#     ss_res = np.sum((y_true - y_pred) ** 2)
#
#     # 计算总方差
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#
#     # 计算R²
#     r2 = 1 - (ss_res / ss_tot)
#
#     return r2
# # expect_force=sine_wave(0,11,0.2,10,100)
# # expect_force=sine_wave(0,11,0.2,10,100)
# # 读取CSV文件，假设文件名为'data.csv'
# data = pd.read_csv('suiji_ring_varify_1_alldata.csv')
# expect_force=data.iloc[:, 3]
# expect_force=[x / 1 for x in expect_force]
# # print(expect_force)
# # 选择前7列作为原始数据，假设它们位于第1到第7列
# raw_data = data.iloc[:, 0:2]
# # raw_data = [x / 10 for x in raw_data]
# # raw_data = raw_data / 10
#
# # 选择第8列作为均值数据，假设均值数据位于第8列
# mean = data.iloc[:, 2]
# mean = [x / 1 for x in mean]
# array1 = np.array(expect_force)
# array2 = np.array(mean)
# r2_value = calculate_r2(array2, array1)
# print(f"R²值: {r2_value}")
# rmse = np.sqrt(np.mean((array2 - array1) ** 2))
# # 计算7列原始数据的标准差，作为偏差范围
# std_dev = raw_data.std(axis=1)
# # print(std_dev[600:700])
# # std_dev[600:700] = [x + 1.2 for x in std_dev[600:700]]   #ring需要用到这句代码
# # 如果第一列是时间列，可以用作时间轴 t
# # 否则用索引作为时间轴
# # if 'time' in data.columns:
# #     t = data['time']  # 假设时间列名为'time'
# # else:
# #     t = np.arange(len(data))  # 使用索引作为时间
# t = np.linspace(0, 10, len(data))  # 生成从0到10的等间距时间轴
# # print(len(t))
# # print(len(mean))
# # print(len(expect_force))
# # 绘制均值线
# plt.figure(figsize=(8, 6))
# plt.plot(t,mean, color='darkorange',alpha=1,linewidth=2, label='Mean')
# plt.plot(t,expect_force,color='black',alpha=0.8, linewidth=2,label='Expected Force',linestyle='--')
# # 绘制阴影表示偏差，±1标准差
# plt.fill_between(t,mean - std_dev, mean + std_dev, color='peachpuff', alpha=0.5)
# plt.text(0.4, 0.7, f'RMSE: {rmse:.2f}', horizontalalignment='center',
#          verticalalignment='center', transform=plt.gca().transAxes, fontsize=18)
# # 图像配置
# plt.title('Irregular/Ring', fontsize=18)
# plt.xlabel('Time (s)', fontsize=15)
# plt.ylabel('Force (N)', fontsize=15)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(loc='upper right',fontsize=15)
# # plt.grid(True)
# plt.show()



# #---------------------------------------------------------------------------------------------------画出阴影偏差----相同期望力
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# def sine_wave(min_value, max_value, frequency, duration, sampling_rate, shift_fraction=0.0):
#     amplitude = (max_value - min_value) / 2.0
#     offset = (max_value + min_value) / 2.0
#     expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#     expect_force_waveform = amplitude * np.sin(
#         2 * np.pi * frequency * (expect_force_t + shift_fraction * duration) - np.pi / 2) + offset
#     return expect_force_waveform
# def triangle_wave(min_value, max_value, period, duration, sampling_rate, shift_fraction=1.0):
#     amplitude = (max_value - min_value) / 2.0
#     offset = (max_value + min_value) / 2.0
#     expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#
#     # 使用 scipy.signal.sawtooth 来生成三角波，triangle 参数设置为 0.5 生成对称三角波
#     expect_force_waveform = amplitude * signal.sawtooth(2 * np.pi * (expect_force_t + shift_fraction * period) / period,0.5) + offset
#     return expect_force_waveform
# # expect_force=sine_wave(0,11,0.2,10,100)
# # expect_force=sine_wave(0,11,0.2,10,100)
# # 读取CSV文件，假设文件名为'data.csv'
# data = pd.read_csv('sine_ring_thumb_same.csv')
# expect_force=data.iloc[:, 12]
# # expect_force=[x / 1 for x in expect_force]
# # 选择前7列作为原始数据，假设它们位于第1到第7列
# raw_data_ring = data.iloc[:, 0:4]
# raw_data_thumb = data.iloc[:, 5:9]
# # raw_data = [x / 10 for x in raw_data]
# # raw_data = raw_data / 10
#
# # 选择第8列作为均值数据，假设均值数据位于第8列
# mean_ring = data.iloc[:, 10]
# mean_thumb = data.iloc[:, 11]
# # mean = [x / 10 for x in mean]
# # 计算7列原始数据的标准差，作为偏差范围
# std_dev_ring = raw_data_ring.std(axis=1)
# std_dev_thumb = raw_data_thumb.std(axis=1)
#
# # 如果第一列是时间列，可以用作时间轴 t
# # 否则用索引作为时间轴
# # if 'time' in data.columns:
# #     t = data['time']  # 假设时间列名为'time'
# # else:
# #     t = np.arange(len(data))  # 使用索引作为时间
# t = np.linspace(0, 10, len(data))  # 生成从0到10的等间距时间轴
# # 绘制均值线
# plt.figure(figsize=(8, 6))
# plt.plot(t,mean_ring, color='darkorange',alpha=1,linewidth=2, label='Mean_ring')
# plt.plot(t,mean_thumb, color='deepskyblue',alpha=1,linewidth=2, label='Mean_thumb')
# plt.plot(t,expect_force,color='black',alpha=0.8, linewidth=2,label='Expected Force',linestyle='--')
# # 绘制阴影表示偏差，±1标准差
# plt.fill_between(t,mean_ring - std_dev_ring, mean_ring + std_dev_ring, color='peachpuff', alpha=0.5)
# plt.fill_between(t,mean_thumb - std_dev_thumb, mean_thumb + std_dev_thumb, color='lightskyblue', alpha=0.5)
#
# # 图像配置
# # plt.title('Random/Thumb', fontsize=14)
# plt.xlabel('Time(s)', fontsize=14)
# plt.ylabel('Force(N)', fontsize=14)
#
# plt.legend()
# # plt.grid(True)
# plt.show()


#---------------------------------------------------------------------------------------------------画出阴影偏差----不同期望力
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# def sine_wave(min_value, max_value, frequency, duration, sampling_rate, shift_fraction=0.0):
#     amplitude = (max_value - min_value) / 2.0
#     offset = (max_value + min_value) / 2.0
#     expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#     expect_force_waveform = amplitude * np.sin(
#         2 * np.pi * frequency * (expect_force_t + shift_fraction * duration) - np.pi / 2) + offset
#     return expect_force_waveform
# def triangle_wave(min_value, max_value, period, duration, sampling_rate, shift_fraction=1.0):
#     amplitude = (max_value - min_value) / 2.0
#     offset = (max_value + min_value) / 2.0
#     expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#
#     # 使用 scipy.signal.sawtooth 来生成三角波，triangle 参数设置为 0.5 生成对称三角波
#     expect_force_waveform = amplitude * signal.sawtooth(2 * np.pi * (expect_force_t + shift_fraction * period) / period,0.5) + offset
#     return expect_force_waveform
# # expect_force=sine_wave(0,11,0.2,10,100)
# # expect_force=sine_wave(0,11,0.2,10,100)
# # 读取CSV文件，假设文件名为'data.csv'
# data = pd.read_csv('sine_ring_thumb_nosame.csv')
# expect_force_ring=data.iloc[:, 11]
# expect_force_thumb=data.iloc[:, 12]
# # expect_force=[x / 1 for x in expect_force]
# # 选择前7列作为原始数据，假设它们位于第1到第7列
# raw_data_ring = data.iloc[:, 0:3]
# raw_data_thumb = data.iloc[:, 4:8]
# # raw_data = [x / 10 for x in raw_data]
# # raw_data = raw_data / 10
#
# # 选择第8列作为均值数据，假设均值数据位于第8列
# mean_ring = data.iloc[:, 9]
# mean_thumb = data.iloc[:, 10]
# # mean = [x / 10 for x in mean]
# # 计算7列原始数据的标准差，作为偏差范围
# std_dev_ring = raw_data_ring.std(axis=1)
# std_dev_thumb = raw_data_thumb.std(axis=1)
#
# # 如果第一列是时间列，可以用作时间轴 t
# # 否则用索引作为时间轴
# # if 'time' in data.columns:
# #     t = data['time']  # 假设时间列名为'time'
# # else:
# #     t = np.arange(len(data))  # 使用索引作为时间
# t = np.linspace(0, 10, len(data))  # 生成从0到10的等间距时间轴
# # 绘制均值线
# plt.figure(figsize=(8, 6))
# plt.plot(t,mean_ring, color='darkorange',alpha=1,linewidth=2, label='Mean ring')
# plt.plot(t,mean_thumb, color='deepskyblue',alpha=1,linewidth=2, label='Mean thumb')
# plt.plot(t,expect_force_ring,color='black',alpha=0.8, linewidth=2,label='Expected ring Force',linestyle='--')
# plt.plot(t,expect_force_thumb,color='black',alpha=0.8, linewidth=2,label='Expected thumb Force',linestyle='--')
# # 绘制阴影表示偏差，±1标准差
# plt.fill_between(t,mean_ring - std_dev_ring, mean_ring + std_dev_ring, color='peachpuff', alpha=0.5)
# plt.fill_between(t,mean_thumb - std_dev_thumb, mean_thumb + std_dev_thumb, color='lightskyblue', alpha=0.5)
#
# # 图像配置
# # plt.title('Random/Thumb', fontsize=14)
# plt.xlabel('Time(s)', fontsize=14)
# plt.ylabel('Force(N)', fontsize=14)
#
# plt.legend()
# # plt.grid(True)
# plt.show()


#----------------------------------------简单画图代码
# import pandas as pd
# import matplotlib.pyplot as plt
#
# df = pd.read_csv('suiji_ring_varify_1_alldata.csv')
#
# # 选择你想要绘制的两列数据，假设为第1列和第2列（索引从0开始）
# # x = df.iloc[:, 0]  # 第一列数据
# y = df.iloc[:, 1]  # 第二列数据
# z = df.iloc[:, 2]
# # z1= df.iloc[:, 3]
# plt.figure()
# # 绘制图表
# # plt.plot(x,label='ring', color='blue', alpha=0.7)
# plt.plot(y,label='thumb', color='r', alpha=0.7)
# plt.plot(z,label='expect_ring', color='green', alpha=0.7)
# # plt.plot(z1,label='expect_thumb', color='green', alpha=0.7)
#
# # 设置图表标题和坐标轴标签
# plt.title('Plot of Two Columns from CSV')
# plt.xlabel('Column 1')
# plt.ylabel('Column 2')
#
# # 显示图例
# plt.legend()
#
# # 显示图表
# plt.show()

# #  ------------------------------截取数据重采样
# import pandas as pd
# import numpy as np
# from scipy.interpolate import interp1d
#
# # 读取CSV文件
# df = pd.read_csv('suiji_ring_varify_8.csv')
#
# # 提取第一列数据
# data = df.iloc[:, 0]
#
# # 截取从第120个到第870个数据点（索引从0开始）
# filtered_data = data.iloc[43:459].reset_index(drop=True)  # 包含870的值
#
# # 创建原始数据的X值
# x_original = np.arange(len(filtered_data))
#
# # 重采样1000个点
# x_resampled = np.linspace(0, len(filtered_data) - 1, 1000)
#
# # 使用线性插值
# interpolator = interp1d(x_original, filtered_data, kind='linear')
# resampled_data = interpolator(x_resampled)
#
# # 创建新的DataFrame
# resampled_df = pd.DataFrame(resampled_data, columns=['Resampled Data'])
#
# # 保存到新的CSV文件
# resampled_df.to_csv('resampled_data_thumb5.csv', index=False)