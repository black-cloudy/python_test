import serial
import time
import threading
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

# --------------------------------------------以下为电刺激串口通信
ser = serial.Serial(port='COM6', baudrate='115200')


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
        writen = ser.write(x)
        # print("current 0")
        return
    elif (applied_current != 0):
        high, low = currentTohigh_low(applied_current)
        x = [0x55, 0x02, channel, 0x01, high, low, 0x00, 0x00, 0x00, 0x0D]
        writen = ser.write(x)
        # print("current applied")
        return
    else:  # just make sure
        high, low = currentTohigh_low(applied_current)
        x = [0x55, 0x00, channel, 0x01, high, low, 0x00, 0x00, 0x00, 0x0D]
        writen = ser.write(x)
        # print("current 0 other situations")
        return


class apply_current_record_force:
    def __init__(self,channel):
        self.port = 'com15'
        self.ser1 = serial.Serial(
            port=self.port,
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS, timeout=1e-2)
        time.sleep(1e-2)  # sleep 100 ms
        self.init_f = [0, 0, 0]
        self.fz = 0
        self.current_list = []
        self.channel = channel
        self.flag=0

    def init_force(self):
        fx_init = 0
        fy_init = 0
        fz_init = 0
        j = 0
        for i in range(20):
            set_update_rate = "020201".encode('utf-8')
            self.ser1.write(set_update_rate)
            dataaa = self.ser1.read(1000).decode('utf-8')
            fx_init = int(dataaa[4:8], 16) + fx_init
            fy_init = int(dataaa[8:12], 16) + fy_init
            fz_init = int(dataaa[12:16], 16) + fz_init
            j = j + 1
        fx_init = fx_init / j
        fy_init = fy_init / j
        fz_init = fz_init / j
        self.init_f[0:3] = fx_init, fy_init, fz_init

    def generate_current_list(self,current_min, current_max):
        while current_min < current_max:
            self.current_list.append(current_min)
            current_min += 0.01

    # 生成正弦波
    def sine_wave(self,min_value, max_value, frequency, duration, sampling_rate, shift_fraction=0.0):
        amplitude = (max_value - min_value) / 2.0
        offset = (max_value + min_value) / 2.0
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        waveform = amplitude * np.sin(2 * np.pi * frequency * (t + shift_fraction * duration) - np.pi / 2) + offset
        self.current_list=waveform

    # 生成三角波
    def triangle_wave(self,min_value, max_value, period, duration, sampling_rate, shift_fraction=0.0):
        amplitude = (max_value - min_value) / 2.0
        offset = (max_value + min_value) / 2.0
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        waveform = amplitude * 2 * np.abs(np.arcsin(np.sin(2 * np.pi * (t + shift_fraction * duration) / period))) / np.pi + offset
        self.current_list=waveform

    def plot_function(self):
        plt.figure()
        # 绘制 电流 随时间变化的曲线
        plt.plot(self.current_list, label='current vs time', color='b', marker='o', markersize=2)
        # 添加图例
        plt.legend()
        # 添加标题和轴标签
        plt.title('current vs time')
        plt.xlabel('Time (s)')
        plt.ylabel('Curent Value')
        # 显示图形
        plt.show()
        time.sleep(1)
    # ---------------------------------力传感器采集数据线程
    def force_threading(self):
        force_list = []
        i = fz_sum = 0
        print('begin')
        while True:
            while i < 2:
                set_update_rate = "020201".encode('utf-8')
                self.ser1.write(bytearray(set_update_rate))
                dataaa = self.ser1.read(1000).decode('utf-8')
                self.fz = int(dataaa[12:16], 16) - self.init_f[2]
                fz_sum = fz_sum + self.fz
                i += 1
            self.fz = fz_sum / 20
            print('测得力：',self.fz )
            force_list.append(self.fz)
            i = fz_sum = 0
            if self.flag >= len(self.current_list):
                break
        # 创建图表
        plt.figure(figsize=(10, 4))

        plt.plot(force_list, marker='o', linestyle='-', color='b')

        # 添加标题和标签
        plt.title('Force Values Over Time')
        plt.xlabel('Time (arbitrary units)')
        plt.ylabel('Force (N)')
        # 显示网格
        plt.grid(True)
        # 显示图表
        plt.tight_layout()
        # 设置横坐标时间限制在0到23
        # plt.xlim(0, 23)
        plt.show()

    # -----------------------------------电刺激线程
    def fes_threading(self):
        data_list = []  # 用来存储电流大小，力大小，刺激时间
        while self.flag < len(self.current_list):
            apply_current_stimulation(self.channel, self.current_list[self.flag])
            # print('通道电流：',self.channel, self.current_list[self.flag])
            time.sleep(0.01)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            current_force_list = [self.current_list[self.flag], self.fz, current_time]
            data_list.append(current_force_list)
            self.flag += 1
        self.flag += 5    # 确保力传感器线程关闭
        apply_current_stimulation(self.channel, 0)  # 刺激结束，将电流置零
        time.sleep(0.08)

        # 指定.csv文件的路径和文件名
        csv_file = './sin_current_force_2_5.csv'                 # 保存文件路径、名称
        # 使用'w'模式打开.csv文件，newline=''用于避免写入空行
        with open(csv_file, 'w', newline='') as file:
            # 创建csv写入对象
            writer = csv.writer(file)
            # 逐行写入数据
            writer.writerows(data_list)
        print("数据已保存到", csv_file)

    def two_threading_task(self):
        fes_thread = threading.Thread(target=self.fes_threading)  # fes线程
        fes_thread.start()

        force_thread = threading.Thread(target=self.force_threading)  # 力传感器线程
        force_thread.start()


        # 等待线程结束
        fes_thread.join()
        force_thread.join()


def main():
    shili=apply_current_record_force(4)     # 输入通道
    shili.sine_wave(3,14,0.2,10,100)   # sin波的最小值、最大值、频率、持续时间、采样率
    shili.plot_function()
    shili.init_force()
    shili.two_threading_task()


if __name__ == '__main__':
    main()

