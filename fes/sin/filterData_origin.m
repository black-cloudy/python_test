% clc;
% clear;
% close all;
% 
% filename = './1_sin_test_2.5.csv';
% ch = 1;
% 
% if ch==1
%     no = 2;
% elseif ch==2
%     no= 18;
% end
% 
% 
% data = readtable(filename);
% pData = data(:, no);
% pData = table2array(pData);
% Fs = 1.259259e+003;
% 
% startIndex = [5000, 10000, 15000, 20000, 25000, 30000];
% 
% low_cutoff = 20;
% high_cutoff = 300;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'bandpass');
% 
% filtered_signal = filtfilt(b, a, pData);
% 
% low_cutoff = 90;
% high_cutoff = 110;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% 
% filtered_signal_notch = filter(b, a, filtered_signal);
% 
% 
% low_cutoff = 190;
% high_cutoff = 210;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% 
% filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% 
% low_cutoff = 290;
% high_cutoff = 310;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% 
% filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% low_cutoff = 258;
% high_cutoff = 262;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% 
% filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% low_cutoff = 390;
% high_cutoff = 410;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% 
% filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% window_size = 200;
% keep_indices = true(size(filtered_signal_notch));
% endIndex = startIndex + 7500;
% startIndex = [startIndex, endIndex];
% for idx = startIndex
%     lower_bound = max(1, idx - window_size);
%     upper_bound = min(length(filtered_signal_notch), idx + window_size);
%     keep_indices(lower_bound:upper_bound) = false;
% end
% filtered_signal_notch = filtered_signal_notch(keep_indices);
% 
% n = length(pData); 
% 
% original_indices = linspace(1, length(filtered_signal_notch), length(filtered_signal_notch));
% 
% new_indices = linspace(1, length(filtered_signal_notch), n);
% 
% filtered_signal_notch = interp1(original_indices, filtered_signal_notch, new_indices, 'linear');
% 
% 
% figure;
% subplot(211)
% plot(pData, 'b');
% title('原始信号');
% xlabel('时间 (s)');
% ylabel('幅度');
% legend('原始信号');
% 
% subplot(212);
% plot(filtered_signal_notch, 'r');
% title('滤波后的信号');
% xlabel('时间 (s)');
% ylabel('幅度');
% legend('滤波后的信号');
% 
% 
% 
% % 使用 pwelch 估计功率谱密度
% [Pxx,f] = pwelch(pData,[],[],[],Fs);
% f_subset = f(:);
% Pxx_subset = Pxx(:);
% 
% [Pxx1,f1] = pwelch(filtered_signal_notch,[],[],[],Fs);
% f_subset1 = f1(:);
% Pxx_subset1 = Pxx1(:);
% 
% 
% figure;
% plot(f_subset, Pxx_subset);
% title('原始信号频谱图');
% xlabel('频率 (Hz)');
% ylabel('能量');
% 
% 
% 
% figure;
% plot(f_subset1, Pxx_subset1);
% title('带通+带阻滤波后频谱图');
% xlabel('频率 (Hz)');
% ylabel('能量');
% 
% 

% 需要修改文件名字，重采样数据段，重采样数据保存文件（保存到力文件的第四列）
% 
clc;
clear;
close all;

filename = './1_sin_test_2.5.csv';       % 是否需要修改文件名，这个文件路面存储了有效的肌电信号
ch = 1;

if ch==1
    no = 2;                       % 提取对应时间的肌电信号出来后就放在文件的第一列了
elseif ch==2
    no= 18;
end

data = readtable(filename);
pData = data(:, no);
pData = table2array(pData);
Fs = 1259;

% 放大信号
amplifiedEMG = pData * 5000;

% 四阶巴特沃斯带通滤波器，频率范围10-450 Hz
[b, a] = butter(4, [10 450]/(Fs/2), 'bandpass');
bandPassedEMG = filter(b, a, amplifiedEMG);

% 归一化处理
% 假设已经有一个变量 MVC 表示最大自愿收缩的电平
% normalizedEMG = bandPassedEMG / 10;                                         %1111111111111111111111111
% 
% % 全波整流
% rectifiedEMG = abs(normalizedEMG);
% 
% threshold_max = 1.08311;                                                    %111111111111111111111111阈值修改
% rectifiedEMG(rectifiedEMG > threshold_max) = NaN;
% % 插值处理NaN值
% rectifiedEMG = fillmissing(rectifiedEMG, 'linear');

% 二阶巴特沃斯低通滤波器，截止频率8 Hz
[d, c] = butter(2, 100/(Fs/2), 'low');
envelopeEMG = filter(d, c, bandPassedEMG);
envelopeEMG=envelopeEMG/5000;


% ----------------计算包络线
windowSize = 400;               % 平滑窗口长度
[up, low] = envelope(envelopeEMG, windowSize, 'peak'); % 使用 'peak' 方法计算包络线
smooth_up = smoothdata(up, 'movmean', windowSize);

% ----------------设置新的采样点数量
newNumPoints = 1000;                                                         % 设置为你需要的新的数据点数量,对应电流和力的数量                                                                    
startIndex =4774;                                                          % 确定要重采样的数据段
endIndex =28858;
% 提取要重采样的数据段
segmentToResample = smooth_up(startIndex:endIndex);
% 原始数据段的索引
originalIndices = linspace(1, length(segmentToResample), length(segmentToResample));
% 新的目标索引，根据需要调整新的数据点数量
newIndices = linspace(1, length(segmentToResample), newNumPoints);
% 进行线性插值重采样
resampledSegment = interp1(originalIndices, segmentToResample, newIndices, 'linear');

% % 读取现有文件
% existingFileName = 'E:\FJY\2024_8_17\1\sin\sin_current_force_2_5.csv';      % 将数据存储到了python计算力文件的路径去了，并且将重采样的肌电信号放在了力-电流文件的第四列
% existingData = readtable(existingFileName);
% % 将重采样后的数据插入到第四列
% if size(existingData, 1) >= newNumPoints
%     existingData(1:newNumPoints, 4) = array2table(resampledSegment');       % 4代表放在第4列  电流、力、时间、肌电信号
% else
%     % 如果现有数据行数不足新数据点数量，添加新行
%     new_data = array2table(resampledSegment', 'VariableNames', existingData.Properties.VariableNames(2));
%     existingData = [existingData; new_data];
% end
% % 将更新后的数据保存回文件
% writetable(existingData, existingFileName);
% % 将重采样后的数据保存到 .csv 文件
% outputFileName = 'resampled_data_column.csv';
% writematrix(resampledSegment', outputFileName);  % 保存为一列

% 绘制结果
figure;
subplot(5,1,1);
plot(pData);
title('原始EMG信号');

% subplot(5,1,2);
% plot(amplifiedEMG);
% title('放大的EMG信号');

subplot(5,1,2);
plot(bandPassedEMG);
title('带通滤波后的EMG信号');

subplot(5,1,3);
plot(envelopeEMG);
title('全波整流后的EMG信号');

subplot(5,1,4);
plot(smooth_up,'g','LineWidth',1.5);
title('信号的包络');
xlabel('样本编号');

subplot(5,1,5);
plot(newIndices, resampledSegment, 'r', 'LineWidth', 1.5);
title('重采样后的数据段');
xlabel('样本索引');
ylabel('幅度');
grid on;

% 确保所有绘图正确显示
drawnow;

