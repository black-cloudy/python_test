# import numpy as np
# import matplotlib.pyplot as plt
#
# # 生成示例EMG信号
# np.random.seed(0)
# time = np.linspace(0, 10, 1000)
# emg_signal = np.sin(2 * np.pi * time) + np.random.normal(0, 0.5, len(time))
#
# # 添加一些突发噪声
# emg_signal[200:210] += 3
# emg_signal[600:610] -= 3
#
# # 移动平均滤波
# window_size = 10
# filtered_signal = np.convolve(emg_signal, np.ones(window_size)/window_size, mode='same')
#
# # 绘图
# plt.figure(figsize=(14, 6))
# plt.subplot(2, 1, 1)
# plt.plot(time, emg_signal, label='Original Signal')
# plt.title('Original EMG Signal')
# plt.subplot(2, 1, 2)
# plt.plot(time, filtered_signal, label='Filtered Signal', color='orange')
# plt.title('EMG Signal after Moving Average Filtering')
# plt.tight_layout()
# plt.show()
#

import numpy as np
import matplotlib.pyplot as plt

# 生成示例EMG信号
np.random.seed(0)
time = np.linspace(0, 10, 1000)
emg_signal = np.sin(2 * np.pi * time) + np.random.normal(0, 0.5, len(time))

# 添加一些突发噪声
emg_signal[200:210] += 3
emg_signal[600:610] -= 3

# 检测伪迹（超过标准差的3倍）
std_dev = np.std(emg_signal)
threshold = 3 * std_dev
artifact_indices = np.where(np.abs(emg_signal - np.mean(emg_signal)) > threshold)[0]

# 用线性插值替换伪迹点
cleaned_signal = np.copy(emg_signal)
cleaned_signal[artifact_indices] = np.nan
cleaned_signal = np.interp(time, time[~np.isnan(cleaned_signal)], cleaned_signal[~np.isnan(cleaned_signal)])

# 平滑处理
window_size = 10
smoothed_signal = np.convolve(cleaned_signal, np.ones(window_size)/window_size, mode='same')

# 绘图
plt.figure(figsize=(14, 6))
plt.subplot(3, 1, 1)
plt.plot(time, emg_signal, label='Original Signal')
plt.title('Original EMG Signal')
plt.subplot(3, 1, 2)
plt.plot(time, cleaned_signal, label='Artifact Removed Signal', color='green')
plt.title('EMG Signal after Artifact Removal')
plt.subplot(3, 1, 3)
plt.plot(time, smoothed_signal, label='Smoothed Signal', color='orange')
plt.title('EMG Signal after Smoothing')
plt.tight_layout()
plt.show()
