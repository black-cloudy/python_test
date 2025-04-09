import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import serial
import time
import csv
import threading
from scipy import signal

# --------------------------------------------以下为电刺激串口通信
ser_current = serial.Serial(port='COM6', baudrate='115200')
def apply_current_stimulation(channel: "int", applied_current: "float"):
    """
    :param channel: 刺激的通道
    :param applied_current: 刺激的电流，0则是关闭
    :return: NULL
    """
    # print(f"channel: {channel}, applied_current: {applied_current}")
    if applied_current > 15:
        print(f"!!!applied_current too much: {applied_current}")
        applied_current = 15

    def currentTohigh_low(current: "float"):
        """
        将刺激的电流转化为两位二进制用以输入
        :param current: 刺激的电流
        :return:
        """
        current = int(4095 * 150 * current / 3300)
        # print(f"n = {current}")
        # print('16: ' + hex(int(current)))
        high = current >> 8
        low = current & 0xff
        # print(f"high = {high}, low = {low}")
        return high, low

    if (applied_current == 0):
        x = [0x55, 0x02, channel, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0D]
        writen = ser_current.write(x)
        # print("current 0")
        return
    elif (applied_current != 0):
        high, low = currentTohigh_low(applied_current)
        x = [0x55, 0x02, channel, 0x01, high, low, 0x00, 0x00, 0x00, 0x0D]
        writen = ser_current.write(x)
        # print("current applied")
        return
    else:  # just make sure
        high, low = currentTohigh_low(applied_current)
        x = [0x55, 0x00, channel, 0x01, high, low, 0x00, 0x00, 0x00, 0x0D]
        writen = ser_current.write(x)
        # print("current 0 other situations")
        return

class force_init:
    def __init__(self,port):
        self.port=port
        self.ser1 = serial.Serial(
            port=self.port,
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS, timeout=1e-2)
        time.sleep(1e-2)  # sleep 100 ms
    # -------------------------------------------力传感器初始化
    def init_force(self):
        # print(ser)
        fx_init = 0
        fy_init = 0
        fz_init = 0
        j = 0
        init_f = []
        for i in range(20):
            set_update_rate = "020201".encode('utf-8')
            # print(type(set_update_rate))
            self.ser1.write(set_update_rate)
            dataaa = self.ser1.read(1000).decode('utf-8')
            fx_init = int(dataaa[4:8], 16) + fx_init
            fy_init = int(dataaa[8:12],16) + fy_init
            fz_init = int(dataaa[12:16],16) + fz_init
            j = j + 1
        fx_init = fx_init / j
        fy_init = fy_init / j
        fz_init = fz_init / j
        init_f[0:3] = fx_init, fy_init, fz_init
        return self.ser1, init_f
class hill_model_nihe:
    def __init__(self,current_force_time_emg_file):
        self.current_force_time_emg_file = current_force_time_emg_file
        # 读取整个CSV文件
        data = pd.read_csv(self.current_force_time_emg_file)
        self.emg_signal = data.iloc[:, 2]
        self.true_force = data.iloc[:, 1]
        self.g1 = 0.000150  # g1（最大电刺激电流下的最大稳定水平值）
        self.g3 = 0.000050  # g3(曲线拐点)是通过sigmode函数图像得到的
        self.Imax = 8  # 最大忍受电流
        self.OA = 0.1  # O为关节点，A为肌腱单元起点，B为插入点
        self.OB = 0.1
        self.lt = 0.2  # 肌腱长度
        self.l0 = 0.06  # 计算最佳肌肉纤维长度的公式系数，yyj的MATLAB代码用的是0.057
        self.ren_mu_da = 0.15  # 计算最佳肌肉纤维长度
        self.l_m0_xishu = 0.1  # 计算最佳肌肉收缩速度的系数(设它为线性的)
        self.lm_fixed = 0.03
        self.cos_phi_fixed = 120
        self.angle = 120
        self.f_mx=40
        self.l_joint_tip=0.05

    def calculate_finger_tip_force(self,I,lm_fixed,xishu1,xishu2):
        f_v = 0
        # -------------------------------------------------------------------------------------------
        """"
        计算激活程度a，g1（最大电刺激电流下的最大稳定水平值）、g2(曲线斜率)、g3(曲线拐点)是通过sigmode函数图像得到的
        Imax是最大能忍受的电流，I是实时的电流
        g2是变化量，通过曲线得到值
        """
        # a = (g1 / (1 + math.exp(g2 * (g3 - I / Imax))))
        a = I
        # a=(np.exp(0.5*I*10000-1))/(np.exp(0.05)-1)
        # --------------------------------------------------------------------------------------------
        """
        l_mt为肌腱和肌肉在肌腱方向的总长，r_mt为力臂，O为关节点，A为肌腱单元起点，B为插入点
        q为OA和OB的夹角，保持变化
        """
        q = math.radians(self.angle)  # 计算cos q，要先把角度值转换为弧度制
        l_mt = math.sqrt(self.OA * self.OA + self.OB * self.OB - 2 * self.OA * self.OB * (math.cos(q)))
        r_mt = (1 / l_mt) * self.OA * self.OB * math.sin(q)
        # -----------------------------------------------------------------------------------------------
        """
        此部分是为了计算归一化肌纤维长度和归一化肌肉收缩速度
        cos_phi为肌肉纤维和肌腱之间的夹角，先设置lm为一个固定值，即初始值，得到phi与q的关系
        lm为肌肉纤维的长度，先设置cos_phi为固定值，即初始值，得到lm与q的关系
        lt是肌腱的长度，默认为定值
        l_m0为最佳肌肉纤维长度，用来归一化肌纤维长度
        """
        cos_phi = (l_mt - self.lt) / lm_fixed  # 得到phi与q之间的关系
        lm = (l_mt - self.lt) / self.cos_phi_fixed  # 得到lm与q之间的关系
        l_m0 = self.l0 * (self.ren_mu_da * (1 - a) + 1)  # 最佳肌肉纤维长度
        l = lm / l_m0  # 归一化肌肉纤维长度
        lm_derivative_q = r_mt / self.cos_phi_fixed  # 得到当前肌肉收缩速度，用长度对角度的求导
        l_m0_derivative_q = l_m0 * self.l_m0_xishu  # 得到当前最佳肌肉收缩速度，用乘以一个系数来表示他的导数
        v = lm_derivative_q / l_m0_derivative_q  # 得到归一化速度

        # ------------------------------------------------------------------------------------------------
        """
        此部分是为了计算指尖的力:
        f_v是计算f(v),f_l是计算f(l),f_mx是最大等长肌力，通过拟合得到的,f_ce是计算肌肉纤维的力
        f_pe是计算并联的力
        f_mt是整个肌肉输出的力,计算出关节输出的力矩，根据力矩和假设关节到指尖为刚性的，计算出指尖输出的力
        """
        if v > 0:
            f_v = (2.34 + 0.039) / (1.3 * v + 0.039)
        if v < 0:
            f_v = 0.3 * (v + 1) / (-v + 0.3)
        f_l = math.sin(-1.317 * l * l - 0.403 * l + 2.545)
        f_ce = f_v * f_l * self.f_mx * a * 1000000  # 肌肉CE的力

        f_pe = (0.129 * math.exp(4.545 * (l - 1)) - 1) * self.f_mx  # 并联PE的力

        f_mt = (f_ce + f_pe) * cos_phi  # 整个肌肉输出的力
        tau_est = r_mt * f_mt  # 关节输出的力矩
        f_tip = xishu1 * (tau_est * f_mt / self.l_joint_tip) / xishu2

        return f_tip, lm

class force2current:
    def __init__(self,emg_and_current_s_nihe_file,calculate_force_list):
        self.emg2current_nihe_file = emg_and_current_s_nihe_file
        self.calculate_force_list = calculate_force_list
        self.expect_force_t=[]
        self.expect_force_waveform=[]
        data_s = pd.read_csv(self.emg2current_nihe_file)
        # 提取第y一列（拟合重采样的电流）和第二列（拟合重采样的肌电信号）
        self.re_sampling_current_s = data_s.iloc[:, 0]  # 电流
        self.re_sampling_emg_signal_s = data_s.iloc[:, 1]  # 肌电
        self.average_emg=0
        self.average_current = 0

    # 生成sin波
    def sine_wave(self,min_value, max_value, frequency, duration, sampling_rate, shift_fraction=0.0):
        amplitude = (max_value - min_value) / 2.0
        offset = (max_value + min_value) / 2.0
        self.expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        self.expect_force_waveform = amplitude * np.sin(2 * np.pi * frequency * (self.expect_force_t + shift_fraction * duration) - np.pi / 2) + offset
        data_number=len(self.expect_force_waveform)
        return data_number

    # 生成三角波
    # def triangle_wave(self, min_value, max_value, period, duration, sampling_rate, shift_fraction=0.0):
    #     amplitude = (max_value - min_value) / 2.0
    #     offset = (max_value + min_value) / 2.0
    #     self.expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    #     self.expect_force_waveform = amplitude * 2 * np.abs(np.arcsin(np.sin(2 * np.pi * (self.expect_force_t + shift_fraction * duration) / period))) / np.pi + offset
    #     data_number = len(self.expect_force_waveform)
    #     return data_number

    def triangle_wave(self,min_value, max_value, period, duration, sampling_rate, shift_fraction=1.0):
        amplitude = (max_value - min_value) / 2.0
        offset = (max_value + min_value) / 2.0
        self.expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        # 使用 scipy.signal.sawtooth 来生成三角波，triangle 参数设置为 0.5 生成对称三角波
        self.expect_force_waveform = amplitude * signal.sawtooth(2 * np.pi * (self.expect_force_t + shift_fraction * period) / period,0.5) + offset
        return len(self.expect_force_waveform)

    # 生成矩形波
    def generate_square_wave(self,min_val, max_val, frequency, duration, sampling_rate):
        self.expect_force_t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        amplitude = (max_val - min_val) / 2
        offset = (max_val + min_val) / 2
        self.expect_force_waveform = amplitude * np.sign(np.sin(2 * np.pi * frequency * self.expect_force_t)) + offset
        data_number = len(self.expect_force_waveform)
        return data_number

    # 生成随机波形
    def generate_smooth_curve(self, min_val, max_val,num_points, noise_level=0.2, smoothing_window=50, seed=42):
        np.random.seed(seed)  # 保证每次生成的波形相同
        # 混合不同频率和振幅的正弦波
        base_waveform = (np.sin(np.linspace(0, 4 * np.pi, num_points)) +
                         0.5 * np.sin(np.linspace(0, 2 * np.pi, num_points)) +
                         0.2 * np.sin(np.linspace(0, 13 * np.pi, num_points)))
        # 添加随机噪声
        noise = np.random.normal(0, noise_level, num_points)
        # 将噪声加入基础波形
        random_waveform = base_waveform + noise
        # 使用卷积对波形进行平滑处理
        self.expect_force_waveform = np.convolve(random_waveform, np.ones(smoothing_window) / smoothing_window,mode='same')
        # 将平滑后的波形归一化到指定的最小值和最大值之间
        self.expect_force_waveform = (self.expect_force_waveform - np.min(self.expect_force_waveform)) / (
                np.max(self.expect_force_waveform) - np.min(self.expect_force_waveform))
        self.expect_force_waveform = self.expect_force_waveform * (max_val - min_val) + min_val
        return num_points

    def plot_function(self):
        plt.figure()
        # 绘制 电流 随时间变化的曲线
        plt.plot(self.expect_force_waveform, label='force vs time', color='b', marker='o', markersize=2)
        # 添加图例
        plt.legend()
        # 添加标题和轴标签
        plt.title('force vs time')
        plt.xlabel('Time (s)')
        plt.ylabel('force Value')
        # 显示图形
        plt.show()
        time.sleep(0.5)

    def find_closest_emg(self,emg_list,j,xishu):      # 用这个肌电信号来计算的模拟力  找到预期力在计算力中最相似的值，则会对应一个emg，找到此emg对应s型拟合中最相似的emg，则有一个对应的电流
        # 将列表转换为numpy数组
        emg_array = np.array(emg_list)
        force_array = np.array(self.calculate_force_list)
        # 计算每个力值与输入力值的差值
        differences = np.abs(force_array - self.expect_force_waveform[j])
        # 找到最小差值的索引
        min_diff_index = np.where(differences == np.min(differences))
        # 获取对应的EMG值
        closest_emgs = emg_array[min_diff_index]
        # 计算平均值
        self.average_emg = np.mean(closest_emgs)*xishu                                        # 当时量纲不一样才*7，使得量纲比较相似  根据sin波下的肌电信号和线性下的肌电信号大小来确定系数

    def find_closest_current(self):
        # 将列表转换为numpy数组
        current_array = np.array(self.re_sampling_current_s)
        sampling_emg_array = np.array(self.re_sampling_emg_signal_s)
        # 计算每个力值与输入力值的差值
        differences = np.abs(sampling_emg_array - self.average_emg)
        # 找到最小差值的索引
        min_diff_index = np.where(differences == np.min(differences))
        # 获取对应的EMG值
        closest_current = current_array[min_diff_index]
        # 计算平均值
        self.average_current = np.mean(closest_current)
        return self.average_current

class give_current_and_record_force:
    def __init__(self,channel,current,init_ff,ser):
        self.channel = channel
        self.current = current
        self.init_ff = init_ff
        self.fz=0
        self.ser = ser

    # ----------------------------------力传感器线程
    def force_threading(self):
        i=0
        sum=0
        while i<2:
            set_update_rate = "020201".encode('utf-8')
            self.ser.write(bytearray(set_update_rate))
            dataaa = self.ser.read(1000).decode('utf-8')
            self.fz = int(dataaa[12:16], 16) - self.init_ff[2]
            sum=sum+self.fz
            i+=1
        self.fz=sum/20

    # -----------------------------------电刺激线程
    def fes_threading(self):
        apply_current_stimulation(self.channel, self.current)
        print(self.channel, self.current)
        time.sleep(0.01)

    def two_threading_task(self):
        fes_thread = threading.Thread(target=self.fes_threading)  # fes线程
        fes_thread.start()

        force_thread = threading.Thread(target=self.force_threading)  # 力传感器线程
        force_thread.start()

        # 等待线程结束
        fes_thread.join()
        force_thread.join()

        return self.fz

class plot_figures:
    def __init__(self,list1,list2,offset,label1,label2,title):
        self.list1 = list1
        self.list2 = list2
        self.offset = offset
        self.label1 = label1
        self.label2 = label2
        self.title = title
    def plot_function(self):
        x1 = np.linspace(0,10,len(self.list1))  # 真实力
        x2 = np.linspace(0,10,len(self.list2))  # 拟合力
        x2_shifted = x2 + self.offset
        plt.figure(dpi=300)
        # 绘制 true_force 随时间变化的曲线
        plt.plot(x1, self.list1, label=self.label1,color='gray', alpha=0.1, marker='o', markersize=2)  # alpha：调透明度
        # 绘制 emg_signal 随时间变化的曲线
        plt.plot(x2_shifted, self.list2, label=self.label2,color='darkorange', marker='x', markersize=2)
        # 添加图例
        plt.legend(loc='lower center',fontsize=16)
        # 添加标题和轴标签
        plt.title(self.title)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Force(N)', fontsize=14)
        plt.ylim([-5, 32])
        # 显示图形
        plt.show()
        time.sleep(0.2)
    def plot_function_1(self,back_current_list):
        plt.figure(dpi=300)
        # 绘制 emg_signal 随时间变化的曲线
        plt.plot(back_current_list, label='back_current', color='r', marker='x', markersize=2)
        # 添加图例
        plt.legend()
        # 添加标题和轴标签
        plt.title('back_current')
        plt.xlabel('Time (s)')
        plt.ylabel('current')
        # 显示图形
        plt.show()
        time.sleep(0.2)

def extract_and_average(data):
    # 计算列表的长度
    n = len(data)

    if n < 3:
        print("列表长度太短，无法提取中间三分之一的数据")
        return None

    # 计算中间三分之一的起始和结束索引
    start = n // 3
    end = 2 * n // 3

    # 提取中间三分之一的数据
    middle_third = data[start:end]

    # 计算平均值
    if middle_third:
        avg = sum(middle_third) / len(middle_third)
    else:
        print("中间部分为空")
        return None

    # 清空原始列表
    data.clear()

    return avg


def main():
    calculate_force_list_index_finger=[]
    closest_current_list_index_finger=[]
    calculate_force_list_thumb_finger = []
    closest_current_list_thumb_finger = []

    lm_initial=0.03
    i=0
    # 计算食指拟合力的大小
    index_finger_force_calculate=hill_model_nihe('sin_current_force_1.csv')            # 存放电流、力、时间、肌电的文件
    for emg_value in index_finger_force_calculate.emg_signal:
        calculate_force, lm=index_finger_force_calculate.calculate_finger_tip_force(emg_value,lm_initial,2.7*5.5,10**12)   #后面两个参数用来调整大小
        calculate_force_list_index_finger.append(calculate_force)
        lm_initial=lm
    # 将拟合力和真实力画出来（如果拟合结果可以的话将这两句代码注释掉）
    # plot_index_finger_true_nihe_firce=plot_figures(index_finger_force_calculate.true_force,calculate_force_list_index_finger,0,'Ring/True force','Ring/Fitting force',' ')
    # plot_index_finger_true_nihe_firce.plot_function()
    # 根据期望力反推肌电，再反推出电流大小
    index_finger_force_mapping_current=force2current('fitted_data_1.csv',calculate_force_list_index_finger)
    # force_data_number=index_finger_force_mapping_current.generate_smooth_curve(0,4,1000)
    force_data_number = index_finger_force_mapping_current.sine_wave(0, 15, 0.2, 10, 100)
    # force_data_number = index_finger_force_mapping_current.triangle_wave(0,5, 5, 10, 100)
    # index_finger_force_mapping_current.plot_function()  # 画出期望的力的波形
    while i<force_data_number:
        index_finger_force_mapping_current.find_closest_emg(index_finger_force_calculate.emg_signal,i,2)              # 1这个系数是用来调整量纲的
        current=index_finger_force_mapping_current.find_closest_current()-5                                          # 疲劳程度带来的，所以增加几个电流
        closest_current_list_index_finger.append(current)
        i+=1
    # index_finger_force_mapping_current.expect_force_waveform = [x * 0.17 for x in index_finger_force_mapping_current.expect_force_waveform]
    # plot_index_finger_true_nihe_firce.plot_function_1(closest_current_list_index_finger)

    lm_initial = 0.03
    i=0
    # 拇指拟合力的大小
    thumb_finger_force_calculate = hill_model_nihe('sin_current_force_2.csv')
    for emg_value in thumb_finger_force_calculate.emg_signal:
        calculate_force, lm = thumb_finger_force_calculate.calculate_finger_tip_force(emg_value, lm_initial,4.2,10**12)
        calculate_force_list_thumb_finger.append(calculate_force)
        lm_initial = lm
    # 将拟合力和真实力画出来
    plot_thumb_finger_true_nihe_firce = plot_figures(thumb_finger_force_calculate.true_force,calculate_force_list_thumb_finger,0,'Thumb/True force', 'Thumb/Fitting force',' ')
    plot_thumb_finger_true_nihe_firce.plot_function()
    # 根据期望力反推肌电，再反推出电流大小
    thumb_finger_force_mapping_current = force2current('fitted_data_2.csv', calculate_force_list_thumb_finger)
    # force_data_number = thumb_finger_force_mapping_current.generate_smooth_curve(3,25,1000)                               # 根据能产生的力的大小来输入最大最小值
    force_data_number = thumb_finger_force_mapping_current.generate_smooth_curve(3, 10, 1000)
    # force_data_number = thumb_finger_force_mapping_current.triangle_wave(0,10, 5, 10, 100)
    # force_data_number = thumb_finger_force_mapping_current.sine_wave(0,10, 0.2, 10, 100)
    thumb_finger_force_mapping_current.plot_function()      # 画出期望的力的波形
    while i < force_data_number:
        thumb_finger_force_mapping_current.find_closest_emg(thumb_finger_force_calculate.emg_signal, i,1)
        current = thumb_finger_force_mapping_current.find_closest_current()-2                                               # 疲劳程度带来的，所以增加几个电流
        closest_current_list_thumb_finger.append(current)
        i += 1
    # thumb_finger_force_mapping_current.expect_force_waveform = [x * 0.5 for x in thumb_finger_force_mapping_current.expect_force_waveform]
    plot_thumb_finger_true_nihe_firce.plot_function_1(closest_current_list_thumb_finger)

    # 将根据期望力反推得到的的电流再次电刺激
    verify_force_index_finger = []
    verify_force_thumb_finger = []
    force_device_1 = force_init('com15')                     # 力传感器初始化
    ser_device_1,init_ff_1=force_device_1.init_force()
    force_device_2= force_init('com12')
    ser_device_2,init_ff_2=force_device_2.init_force()
    j=0
    while j<len(closest_current_list_thumb_finger):
        # if j<200:
        #     give_index_finger_current=give_current_and_record_force(5,closest_current_list_index_finger[j],init_ff_1,ser_device_1)  # 通道、电流、力传感器初始化的两个参数
        #     verify_index_force=give_index_finger_current.two_threading_task()
        #     print('食指验证力1：',verify_index_force)
        #     verify_force_index_finger.append(verify_index_force)
        # elif j>200 and j<500:
        #     time.sleep(0.2)
        #     give_index_finger_current = give_current_and_record_force(5, closest_current_list_index_finger[j]+4,
        #                                                               init_ff_1, ser_device_1)  # 通道、电流、力传感器初始化的两个参数
        #     verify_index_force = give_index_finger_current.two_threading_task()
        #     print('食指验证力2：', verify_index_force)
        #     verify_force_index_finger.append(verify_index_force)
        # else:
        #     give_index_finger_current = give_current_and_record_force(5, closest_current_list_index_finger[j],
        #                                                               init_ff_1, ser_device_1)  # 通道、电流、力传感器初始化的两个参数
        #     verify_index_force = give_index_finger_current.two_threading_task()
        #     print('食指验证力3：', verify_index_force)
        #     verify_force_index_finger.append(verify_index_force)

        # give_index_finger_current=give_current_and_record_force(3,closest_current_list_index_finger[j],init_ff_1,ser_device_1)  ## 通道、电流、力传感器初始化的两个参数
        # verify_index_force=give_index_finger_current.two_threading_task()
        # print('食指验证力：', verify_index_force)
        # verify_force_index_finger.append(verify_index_force)

        give_thumb_finger_current=give_current_and_record_force(4,closest_current_list_thumb_finger[j],init_ff_2,ser_device_2)  ## 通道、电流、力传感器初始化的两个参数
        verify_thumb_force=give_thumb_finger_current.two_threading_task()
        print('拇指验证力：', verify_thumb_force)
        verify_force_thumb_finger.append(verify_thumb_force)
        j=j+1
    apply_current_stimulation(4, 0)
    time.sleep(0.5)
    plt.figure()
    # 绘制食指期望力随时间变化的曲线
    # plt.plot(index_finger_force_mapping_current.expect_force_waveform, label='index_expect_force', color='b', marker='o', markersize=2)
    # 绘制食指力随时间变化的曲线
    # plt.plot(verify_force_index_finger, label='index_force', color='r', marker='x', markersize=2)
    # 绘制拇指期望力随时间变化的曲线
    plt.plot( thumb_finger_force_mapping_current.expect_force_waveform, label='thumb_expect_force', color='c',marker='o', markersize=2)
    # 绘制拇指力随时间变化的曲线
    plt.plot(verify_force_thumb_finger, label='thumb_force', color='g', marker='_', markersize=2)
    # 添加图例
    plt.legend()
    # 添加标题和轴标签
    plt.title('expect_index_thumb_force_64')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')

    # 显示图形
    plt.show()
    time.sleep(2)

    # 将两个列表写入到CSV文件中
    with open('suiji_thumb_5.csv', 'w', newline='') as file:                                                                   # 存放两个手指力的文件
        writer = csv.writer(file)

        # 使用 zip 将两个列表的对应元素配对
        for item1, item2 in zip(verify_force_thumb_finger, thumb_finger_force_mapping_current.expect_force_waveform):
            writer.writerow([item1, item2])

    print("Data has been written to output.csv")


if __name__ == '__main__':
    main()


