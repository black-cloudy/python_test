
"""
本代码是用来测试电流阈值和找点位
"""

import serial
import time
import threading
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import struct
import numpy as np
import socket
import matplotlib.pyplot as plt
# --------------------------------------------以下为电刺激串口通信
ser1 = serial.Serial(
        # port='/dev/ttyUSB0',
        port='com15',
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS, timeout=1e-2)
time.sleep(1e-2)  # sleep 100 ms                             力传感器通信

ser = serial.Serial(port='COM6', baudrate='115200')         #电刺激通信
def apply_current_stimulation(channel: "int", applied_current: "float"):
    """
    :param channel: 刺激的通道
    :param applied_current: 刺激的电流，0则是关闭
    :return: NULL
    """
    # print(f"channel: {channel}, applied_current: {applied_current}")
    if applied_current > 10:
        print(f"!!!applied_current too much: {applied_current}")
        applied_current = 10

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
        x = [0x55, 0x02, channel, 0x01, high, low, 0x00, 0x00, 0x00, 0x0D]   # 第七位改为0x00表示一直打开该通道
        writen = ser.write(x)
        # print("current applied")
        return
    else:  # just make sure
        high, low = currentTohigh_low(applied_current)
        x = [0x55, 0x00, channel, 0x01, high, low, 0x00, 0x00, 0x00, 0x0D]
        writen = ser.write(x)
        # print("current 0 other situations")
        return
def init_force():
    fx_init = 0
    fy_init = 0
    fz_init = 0
    j = 0
    init_f = []
    for i in range(20):
        set_update_rate = "020201".encode('utf-8')
        print(type(set_update_rate))
        ser1.write(set_update_rate)
        dataaa = ser1.read(1000).decode('utf-8')
        fx_init = int(dataaa[4:8], 16) + fx_init
        fy_init = int(dataaa[8:12],16) + fy_init
        fz_init = int(dataaa[12:16],16) + fz_init
        j = j + 1
    fx_init = fx_init / j
    fy_init = fy_init / j
    fz_init = fz_init / j
    init_f[0:3] = fx_init, fy_init, fz_init
    return init_f

def main_force():
    global init_ff
    print('begin')
    # 设置绘图
    plt.ion()  # 打开交互模式
    # fig, ax = plt.subplots()
    # 初始化 z 数据列表
    z_data = []
    try:
        while True:
            # data = ser.readall()
            set_update_rate = "020201".encode('utf-8')
            ser1.write(bytearray(set_update_rate))
            dataaa = ser1.read(1000).decode('utf-8')
            if len(dataaa) < 8:
                print("Error: dataaa does not have enough characters.")
            else:
                fz = 0.1*(int(dataaa[12:16], 16) - init_ff[2])
                print('11111111111',fz)
                # line, = ax.plot(fz)
                # z_data.append(fz)

                # 更新绘图
                # line.set_ydata(z_data)
                # line.set_xdata(np.arange(len(z_data)))
                # ax.relim()
                # ax.autoscale_view()
                # plt.draw()
                # plt.pause(0.01)  # 暂停以便实时更新
                # np.savetxt('lq8804.csv', z_data, delimiter=',')

    except KeyboardInterrupt:
        # 当用户按下 Ctrl+C 时结束循环
        print("实时绘图结束")


def fes_threading():
    # channel=3
    current=7

    # while current<20:
    #     apply_current_stimulation(channel,current)0

    #     print('测试的通道和电流：',channel,current)
    #     time.sleep(0.5)
    #     current+=0.3
    while True:
        apply_current_stimulation(6, current)   # 通道3  4--11   通道11  2-8
        time.sleep(0.01)
        # print('测试的通道和电流：',current)
        # apply_current_stimulation(11, 5)
        # time.sleep(0.01)
        # current+=0.01


def main(                                              ):
    global init_ff
    init_ff = init_force()

    fes_thread = threading.Thread(target=fes_threading)  # fes线程
    fes_thread.start()

    force_thread = threading.Thread(target=main_force)  # 力传感器线程
    force_thread.start()


if __name__ == '__main__':
    main()

