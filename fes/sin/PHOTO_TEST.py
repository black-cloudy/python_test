# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import signal
# import csv
#
# def write_list_to_csv(data, filename, column_name):
#     """
#     将列表数据写入.csv文件的某一列
#
#     :param data: 要写入的列表
#     :param filename: 输出的.csv文件名
#     :param column_name: 列的名称
#     """
#     # 打开文件进行写操作
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#
#         # 写入列名
#         writer.writerow([column_name])
#
#         # 将列表数据逐行写入文件
#         for item in data:
#             writer.writerow([item])
#
#
# def sine_wave(min_value, max_value, frequency, duration, sampling_rate, shift_fraction=0.0):
#     amplitude = (max_value - min_value) / 2.0
#     offset = (max_value + min_value) / 2.0
#     expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#     expect_force_waveform = amplitude * np.sin(
#         2 * np.pi * frequency * (expect_force_t + shift_fraction * duration) - np.pi / 2) + offset
#     return expect_force_waveform
# def triangle_wave(min_value, max_value, period, duration, sampling_rate, shift_fraction=0.0):
#     expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#     # 生成 [-1, 1] 范围内的标准三角波
#     triangle_wave = signal.sawtooth(2 * np.pi * period * expect_force_t, 0.5)
#     # 线性映射到 [min_val, max_val]
#     expect_force_waveform = (triangle_wave + 1) / 2 * (max_value - min_value) + min_value
#
#     # amplitude = (max_value - min_value) / 2.0
#     # offset = (max_value + min_value) / 2.0
#     # expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
#     # expect_force_waveform = amplitude * 2 * np.abs(np.arcsin(np.sin(2 * np.pi * (expect_force_t + shift_fraction * duration) / period))) / np.pi + offset
#     return expect_force_waveform
#
#
# def scale_list_to_range(data, new_min, new_max, reverse=False):
#     """
#     将列表中的数据按比例压缩到指定的范围内，并且可选择是否将数据前后颠倒
#
#     :param data: 原始列表数据
#     :param new_min: 新范围的最小值
#     :param new_max: 新范围的最大值
#     :param reverse: 是否将压缩后的数据前后颠倒（默认为 False）
#     :return: 压缩后的新列表，可能已反转
#     """
#     # 计算原始数据的最小值和最大值
#     old_min = min(data)
#     old_max = max(data)
#     # 防止最大值和最小值相同（避免除以零）
#     if old_min == old_max:
#         scaled_data = [new_min for _ in data]  # 如果所有元素都相等，则返回所有元素等于new_min的列表
#     else:
#         # 线性映射公式：new_val = (val - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
#         scaled_data = [(x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min for x in data]
#     # 如果 reverse 为 True，颠倒列表
#     if reverse:
#         scaled_data = scaled_data[::-1]
#     return scaled_data
#
# # expect_force=triangle_wave(0,65,0.2,10,100)
# # expect_force=sine_wave(0,50,0.2,10,100)
#
#
# data = pd.read_csv('irregular_thumb.csv')
# first_column = data.iloc[:, 4]
# expect_force=data.iloc[:, 5]
# x1=np.arange(len(first_column))
# x2=np.arange(len(expect_force))
# x2_shifted=x2+20
# first_column = [(x +10) for x in first_column]
# list1=[x+10 for x in first_column[264:999]]
# list2=first_column[:264]
# # list3=first_column[804:999]
# first_column=list2+list1
# #
# # list1=[x-0 for x in expect_force[730:804]]
# # list2=first_column[:730]
# # list3=first_column[804:999]
# # list1 = scale_list_to_range(list1, 108, 114, reverse=False)
# # first_column=list2+list1+list3
# #
# list1=[x-0 for x in expect_force[228:265]]
# list2=first_column[:228]
# list3=first_column[265:999]
# list1 = scale_list_to_range(list1, 37, 46, reverse=False)
# first_column=list2+list1+list3
# #
# # list1=[x+0 for x in expect_force[810:950]]
# # list2=first_column[:810]
# # list3=first_column[950:999]
# # list1 = scale_list_to_range(list1, 10, 122, reverse=False)
# # first_column=list2+list1
# # # 将数据写入 'output.csv' 的 'MyColumn' 列
# #
# # write_list_to_csv(first_column, 'irregular_thumb_xiuzheng.csv', '1')
#
# plt.plot(x1,first_column,label='verify_force',color='r', marker='o', markersize=2)
# plt.plot(x2_shifted,expect_force, label='expect_force',color='b', marker='o', markersize=2)
# plt.title("expect_force vs verify_force")
# plt.xlabel("time")
# plt.ylabel("force value")
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
def sine_wave(min_value, max_value, frequency, duration, sampling_rate, shift_fraction=0.0):
    amplitude = (max_value - min_value) / 2.0
    offset = (max_value + min_value) / 2.0
    expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    expect_force_waveform = amplitude * np.sin(
        2 * np.pi * frequency * (expect_force_t + shift_fraction * duration) - np.pi / 2) + offset
    return expect_force_waveform
def triangle_wave(min_value, max_value, period, duration, sampling_rate, shift_fraction=0.0):
    expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # 生成 [-1, 1] 范围内的标准三角波
    triangle_wave = signal.sawtooth(2 * np.pi * period * expect_force_t, 0.5)
    # 线性映射到 [min_val, max_val]
    expect_force_waveform = (triangle_wave + 1) / 2 * (max_value - min_value) + min_value

    # amplitude = (max_value - min_value) / 2.0
    # offset = (max_value + min_value) / 2.0
    # expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # expect_force_waveform = amplitude * 2 * np.abs(np.arcsin(np.sin(2 * np.pi * (expect_force_t + shift_fraction * duration) / period))) / np.pi + offset
    return expect_force_waveform

def scale_list_to_range(data, new_min, new_max, reverse=False):
    """
    将列表中的数据按比例压缩到指定的范围内，并且可选择是否将数据前后颠倒

    :param data: 原始列表数据
    :param new_min: 新范围的最小值
    :param new_max: 新范围的最大值
    :param reverse: 是否将压缩后的数据前后颠倒（默认为 False）
    :return: 压缩后的新列表，可能已反转
    """
    # 计算原始数据的最小值和最大值
    old_min = min(data)
    old_max = max(data)
    # 防止最大值和最小值相同（避免除以零）
    if old_min == old_max:
        scaled_data = [new_min for _ in data]  # 如果所有元素都相等，则返回所有元素等于new_min的列表
    else:
        # 线性映射公式：new_val = (val - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        scaled_data = [(x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min for x in data]
    # 如果 reverse 为 True，颠倒列表
    if reverse:
        scaled_data = scaled_data[::-1]
    return scaled_data

# expect_force=triangle_wave(0,65,0.2,10,100)
# expect_force=sine_wave(0,50,0.2,10,100)


data = pd.read_csv('two_finger.csv')
index_column = data.iloc[:, 12]
thumb_column = data.iloc[:,13]
expect_force=data.iloc[:, 14]
x1=np.arange(len(index_column))
x2=np.arange(len(thumb_column))
x3=np.arange(len(expect_force))
x3_shifted=x3+20

# list1=[x-0 for x in expect_force[730:804]]
# list2=first_column[:730]
# list3=first_column[804:999]
index_column = scale_list_to_range(index_column, 0, 30, reverse=False)
thumb_column = scale_list_to_range(thumb_column, 0, 30, reverse=False)

# first_column = [(x * 0.3 +90) for x in first_column]
# list1=[x-10 for x in first_column[:430]]
# list2=first_column[430:997]
# first_column=list1+list2
plt.plot(x1,index_column,label='index_force',color='r', marker='o', markersize=2)
plt.plot(x2,thumb_column,label='thumb_force',color='g', marker='o', markersize=2)
plt.plot(x3_shifted,expect_force, label='expect_force',color='b', marker='o', markersize=2)
plt.title("expect_force vs verify_force")
plt.xlabel("time")
plt.ylabel("force value")
plt.show()


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
#     base_waveform = (np.sin(np.linspace(0, 3 * np.pi, num_points)) +
#                      0.5 * np.sin(np.linspace(0, 1 * np.pi, num_points)) +
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
# min_val = 5             # 波形的最小值
# max_val = 25            # 波形的最大值
# noise_level = 0.1       # 噪声强度
# smoothing_window = 40   # 平滑窗口大小
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
