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
normalizedEMG = bandPassedEMG / 10;                                         %1111111111111111111111111

% 全波整流
rectifiedEMG = abs(normalizedEMG);

threshold_max = 1.08311;                                                    %111111111111111111111111阈值修改
rectifiedEMG(rectifiedEMG > threshold_max) = NaN;
% 插值处理NaN值
rectifiedEMG = fillmissing(rectifiedEMG, 'linear');

% 二阶巴特沃斯低通滤波器，截止频率8 Hz
[d, c] = butter(2, 8/(Fs/2), 'low');
envelopeEMG = filter(d, c, rectifiedEMG);
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

% 读取现有文件
existingFileName = 'E:\FJY\2024_8_17\1\sin\sin_current_force_2_5.csv';      % 将数据存储到了python计算力文件的路径去了，并且将重采样的肌电信号放在了力-电流文件的第四列
existingData = readtable(existingFileName);
% 将重采样后的数据插入到第四列
if size(existingData, 1) >= newNumPoints
    existingData(1:newNumPoints, 4) = array2table(resampledSegment');       % 4代表放在第4列  电流、力、时间、肌电信号
else
    % 如果现有数据行数不足新数据点数量，添加新行
    new_data = array2table(resampledSegment', 'VariableNames', existingData.Properties.VariableNames(2));
    existingData = [existingData; new_data];
end
% 将更新后的数据保存回文件
writetable(existingData, existingFileName);
% 将重采样后的数据保存到 .csv 文件
outputFileName = 'resampled_data_column.csv';
writematrix(resampledSegment', outputFileName);  % 保存为一列

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

% % 使用 pwelch 估计功率谱密度
% [Pxx,f] = pwelch(pData,[],[],[],Fs);
% f_subset = f(:);
% Pxx_subset = Pxx(:);
% 
% [Pxx1,f1] = pwelch(bandPassedEMG,[],[],[],Fs);
% f_subset1 = f1(:);
% Pxx_subset1 = Pxx1(:);
% 
% figure;
% plot(f_subset, Pxx_subset);
% title('原始信号频谱图');
% xlabel('频率 (Hz)');
% ylabel('能量');
% 
% figure;
% plot(f_subset1, Pxx_subset1);
% title('带通+带阻滤波后频谱图');
% xlabel('频率 (Hz)');
% ylabel('能量');
% 
% 
% [Pxx2,f2] = pwelch(rectifiedEMG,[],[],[],Fs);
% f_subset2 = f2(:);
% Pxx_subset2 = Pxx2(:);
% 
% figure;
% plot(f_subset2, Pxx_subset2);
% title('带通+带阻滤波后频谱图');
% xlabel('频率 (Hz)');
% ylabel('能量');

% 确保所有绘图正确显示
drawnow;



















% % 需要修改文件名字，重采样数据段，重采样数据保存文件（保存到力文件的第四列）
% 
% clc;
% clear;
% close all;
% 
% filename = './1_sin_test_1.csv';       % 是否需要修改文件名，这个文件路面存储了有效的肌电信号
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
% Fs = 1259;
% 
% % startIndex = [46000, 51050, 56100, 61150, 66200, 71250, 76300, 81350, 86400, 91450, 96500, 101550, 106600, 111650, 116700, 121750, 126800, 131850];
% 
% low_cutoff = 20;
% high_cutoff = 150;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'bandpass');
% 
% filtered_signal = filtfilt(b, a, pData);
% 
% low_cutoff = 45;
% high_cutoff = 55;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% 
% filtered_signal_notch = filter(b, a, filtered_signal);
% 
% low_cutoff = 98;
% high_cutoff = 102;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% 
% filtered_signal_notch = filter(b, a, filtered_signal);
% % 
% % low_cutoff = 190;
% % high_cutoff = 210;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % 
% % filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% 
% % low_cutoff = 290;
% % high_cutoff = 310;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % 
% % filtered_signal_notch = filter(b, a, filtered_signal_notch);
% % 
% % low_cutoff = 40;
% % high_cutoff = 59;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % 
% % filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% % low_cutoff = 155;
% % high_cutoff = 160;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% % low_cutoff = 258;
% % high_cutoff = 262;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % 
% % filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% % low_cutoff = 390;
% % high_cutoff = 410;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % 
% % filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% 
% % low_cutoff = 140;
% % high_cutoff = 160;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% threshold_max = 0.00058;
% filtered_signal_notch(abs(filtered_signal_notch) > threshold_max) = NaN;
% 
% % 插值处理NaN值
% filtered_signal_notch = fillmissing(filtered_signal_notch, 'linear');
% 
% % window_size = 200;
% % keep_indices = true(size(filtered_signal_notch));
% % endIndex = startIndex + 7500;
% % startIndex = [startIndex, endIndex];
% % for idx = startIndex
% %     lower_bound = max(1, idx - window_size);
% %     upper_bound = min(length(filtered_signal_notch), idx + window_size);
% %     keep_indices(lower_bound:upper_bound) = false;
% % end
% % filtered_signal_notch = filtered_signal_notch(keep_indices);
% 
% n = length(pData); 
% 
% original_indices = linspace(1, length(filtered_signal_notch), length(filtered_signal_notch));
% 
% new_indices = linspace(1, length(filtered_signal_notch), n);
% 
% filtered_signal_notch = interp1(original_indices, filtered_signal_notch, new_indices, 'linear');
% 
% % 使用逻辑索引将负值取反
% filtered_signal_notch(filtered_signal_notch < 0) = -filtered_signal_notch(filtered_signal_notch < 0);
% 
% % ----------------计算包络线
% windowSize = 800;               % 平滑窗口长度
% [up, low] = envelope(filtered_signal_notch, windowSize, 'peak'); % 使用 'peak' 方法计算包络线
% smooth_up = smoothdata(up, 'movmean', windowSize);
% 
% 
% % ----------------设置新的采样点数量
% newNumPoints =1000;  % 设置为你需要的新的数据点数量
% 
% % 确定要重采样的数据段
% startIndex =1;
% endIndex = 62900;
% 
% % 提取要重采样的数据段
% segmentToResample = smooth_up(startIndex:endIndex);
% 
% % 原始数据段的索引
% originalIndices = linspace(1, length(segmentToResample), length(segmentToResample));
% 
% % 新的目标索引，根据需要调整新的数据点数量
% newIndices = linspace(1, length(segmentToResample), newNumPoints);
% 
% % 进行线性插值重采样
% resampledSegment = interp1(originalIndices, segmentToResample, newIndices, 'linear');
% 
% 
% % 读取现有文件
% existingFileName = 'E:\FJY\2024_8_17\1\sin\sin_current_force_2.csv';      % 将数据存储到了python计算力文件的路径去了，并且将重采样的肌电信号放在了力-电流文件的第四列
% existingData = readtable(existingFileName);
% % 将重采样后的数据插入到第二列
% if size(existingData, 1) >= newNumPoints
%     existingData(1:newNumPoints, 4) = array2table(resampledSegment');       % 4代表放在第4列
% else
%     % 如果现有数据行数不足新数据点数量，添加新行
%     new_data = array2table(resampledSegment', 'VariableNames', existingData.Properties.VariableNames(2));
%     existingData = [existingData; new_data];
% end
% 
% % 将更新后的数据保存回文件
% writetable(existingData, existingFileName);
% % 将重采样后的数据保存到 .csv 文件
% outputFileName = 'resampled_data_column.csv';
% writematrix(resampledSegment', outputFileName);  % 保存为一列
% 
% subplot(3, 1, 1);
% plot(newIndices, resampledSegment, 'r', 'LineWidth', 1.5);
% title('重采样后的数据段');
% xlabel('样本索引');
% ylabel('幅度');
% grid on;
% 
% figure;
% subplot(211)
% plot(pData, 'b');
% title('原始信号');
% xlabel('时间 (s)');
% ylabel('幅度');
% legend('原始信号','包络线');
% 
% subplot(212);
% % plot(filtered_signal_notch, 'r');
% % hold on;
% plot(smooth_up, 'g', 'LineWidth', 1.5, 'DisplayName', 'Upper Envelope');
% title('滤波后的信号');
% xlabel('时间 (s)');
% ylabel('幅度');
% legend('滤波后的信号','包络线');
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
% figure;
% plot(f_subset, Pxx_subset);
% title('原始信号频谱图');
% xlabel('频率 (Hz)');
% ylabel('能量');
% 
% figure;
% plot(f_subset1, Pxx_subset1);
% title('带通+带阻滤波后频谱图');
% xlabel('频率 (Hz)');
% ylabel('能量');
% 


% 
% clc;
% clear;
% close all;
% 
% filename = './extracted_emg_signal.csv';       % 是否需要修改文件名，这个文件路面存储了有效的肌电信号
% ch = 1;
% 
% if ch==1
%     no = 1;                       % 提取对应时间的肌电信号出来后就放在文件的第一列了
% elseif ch==2
%     no= 18;
% end
% 
% 
% data = readtable(filename);
% pData = data(:, no);
% pData = table2array(pData);
% Fs = 1259;
% 
% % 放大信号
% amplifiedEMG = pData * 5000;
% 
% % 四阶巴特沃斯带通滤波器，频率范围10-450 Hz
% [b, a] = butter(4, [10 450]/(Fs/2), 'bandpass');
% bandPassedEMG = filter(b, a, amplifiedEMG);
% 
% low_cutoff = 40;
% high_cutoff = 60;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% bandPassedEMG  = filter(b, a, bandPassedEMG );
% % 
% % low_cutoff = 90;
% % high_cutoff = 110;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % bandPassedEMG  = filter(b, a, bandPassedEMG );
% % 
% % low_cutoff = 140;
% % high_cutoff = 160;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % bandPassedEMG  = filter(b, a, bandPassedEMG );
% % 
% % low_cutoff = 190;
% % high_cutoff = 210;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % bandPassedEMG  = filter(b, a, bandPassedEMG );
% 
% % low_cutoff = 240;
% % high_cutoff = 260;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % bandPassedEMG  = filter(b, a, bandPassedEMG );
% % 
% % low_cutoff = 290;
% % high_cutoff = 310;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % bandPassedEMG  = filter(b, a, bandPassedEMG );
% % 
% % low_cutoff = 340;
% % high_cutoff = 360;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % bandPassedEMG  = filter(b, a, bandPassedEMG );
% % 
% % low_cutoff = 390;
% % high_cutoff = 410;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % bandPassedEMG  = filter(b, a, bandPassedEMG );
% % 
% % low_cutoff = 440;
% % high_cutoff = 450;
% % [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% % bandPassedEMG  = filter(b, a, bandPassedEMG );
% 
% 
% % 归一化处理
% % 假设已经有一个变量 MVC 表示最大自愿收缩的电平
% normalizedEMG = bandPassedEMG / 8;
% 
% % 全波整流
% rectifiedEMG = abs(normalizedEMG);
% 
% % 二阶巴特沃斯低通滤波器，截止频率8 Hz
% [d, c] = butter(2, 8/(Fs/2), 'low');
% envelopeEMG = filter(d, c, rectifiedEMG);
% 
% % 绘制结果
% figure;
% subplot(5,1,1);
% plot(pData);
% title('原始EMG信号');
% 
% subplot(5,1,2);
% plot(amplifiedEMG);
% title('放大的EMG信号');
% 
% subplot(5,1,3);
% plot(bandPassedEMG);
% title('带通滤波后的EMG信号');
% 
% subplot(5,1,4);
% plot(rectifiedEMG);
% title('全波整流后的EMG信号');
% 
% subplot(5,1,5);
% plot(envelopeEMG);
% title('信号的包络');
% xlabel('样本编号');
% 
% % 使用 pwelch 估计功率谱密度
% [Pxx,f] = pwelch(pData,[],[],[],Fs);
% f_subset = f(:);
% Pxx_subset = Pxx(:);
% 
% [Pxx1,f1] = pwelch(bandPassedEMG,[],[],[],Fs);
% f_subset1 = f1(:);
% Pxx_subset1 = Pxx1(:);
% 
% figure;
% plot(f_subset, Pxx_subset);
% title('原始信号频谱图');
% xlabel('频率 (Hz)');
% ylabel('能量');
% 
% figure;
% plot(f_subset1, Pxx_subset1);
% title('带通+带阻滤波后频谱图');
% xlabel('频率 (Hz)');
% ylabel('能量');
% 
% % 确保所有绘图正确显示
% drawnow;





% 
% % -----------------------------------------------------------------------------------------
% clc;
% clear;
% close all;
% 
% filename = './EMG_1819_3.csv';
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
% Fs = 1259;
% 
% startIndex = [3352, 5850, 8403, 13440, 18477, 23527, 28576, 33626, 38676, 43713, 48763, 53800, 58837, 63887, 68924, 73974, 79024, 84061, 89111, 94160, 99197, 104247, 109297, 114347, 119384, 124434, 129482, 134521];
% startIndex = [21008, 80370];
% startIndex = [23377, 29857, 36337, 42817, 49297, 55777, 62257, 68737, 75217, 81697, 88177, 94657, 101137, 107617, 114097, 120577, 127057, 133537, 140017];      %1626
% 
% low_cutoff = 1;
% high_cutoff = 300;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'bandpass');
% 
% filtered_signal = filtfilt(b, a, pData);
% 
% low_cutoff = 40;
% high_cutoff = 60;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% 
% filtered_signal_notch = filter(b, a, filtered_signal);
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
% low_cutoff = 48;
% high_cutoff = 52;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
% 
% filtered_signal_notch = filter(b, a, filtered_signal_notch);
% 
% low_cutoff = 155;
% high_cutoff = 160;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'stop');
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
% % 手动定义16组数据的开始和结束索引
% startIndex = [43827, 49002, 54134, 58842, 63611, 68402, 73042, 77776, 82497, 87234, 91867, 96642, 101354, 106018, 110797, 115433];
% endIndex = [46473, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 94123, 98840, 103531, 108233, 112981, 117689]; % 自定义结束索引
% startIndex=[21024,130473,253167,376082,27631,]
% endIndex=[26923,135722,258645,381605,]
% 
% 
% % 提取每组数据并存放在矩阵的列中
% numGroups = length(startIndex);
% groupLength = 7500;                       % 假设每组数据长度为7500
% dataMatrix = NaN(groupLength, numGroups); % 初始化矩阵
% 
% for i = 1:numGroups
%     dataSegment = filtered_signal_notch(startIndex(i):endIndex(i));
%     dataMatrix(1:length(dataSegment), i) = dataSegment;
% end
% 
% % 保存数据矩阵为CSV文件
% save_path = './filtered_segments.csv';
% writematrix(dataMatrix, save_path);
% 
% 
% % 指定需要提取的数据段的开始和结束索引
% start_index = 43703; % 示例起始索引
% end_index = 117715; % 示例结束索引
% 
% % 提取数据段
% filtered_segment = filtered_signal_notch(start_index:end_index);
% 
% % 确定每组数据的长度
% num_groups = 32;
% group_length = ceil(length(filtered_segment) / num_groups);
% 
% % 初始化一个矩阵来存储数据，每列对应一个组
% data_matrix = NaN(group_length, num_groups);
% 
% % 将数据填入矩阵中
% for i = 1:num_groups
%     start_idx = (i-1) * group_length + 1;
%     end_idx = min(i * group_length, length(filtered_segment));
% 
%     % 提取当前组的数据
%     current_group = filtered_segment(start_idx:end_idx);
% 
%     % 将当前组的数据填入矩阵的第 i 列
%     data_matrix(1:length(current_group), i) = current_group;
% end
% 
% % 指定文件路径和文件名
% csv_save_path = './filtered_signal_notch_groups.csv';
% 
% % 保存数据矩阵到 CSV 文件
% writematrix(data_matrix, csv_save_path);
% 
% % 验证保存的文件路径
% disp('Filtered signal groups have been saved to:');
% disp(csv_save_path);
% 
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
% 使用 pwelch 估计功率谱密度
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
% ---------------------------------------------------------去除异常值，有包络线------------------------------------
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
% Fs = 1259;
% 
% startIndex = [23377, 29857, 36337, 42817, 49297, 55777, 62257, 68737, 75217, 81697, 88177, 94657, 101137, 107617, 114097, 120577, 127057, 133537, 140017];      %1626
% 
% 
% low_cutoff = 1;
% high_cutoff = 300;
% [b, a] = butter(4, [low_cutoff/(Fs/2), high_cutoff/(Fs/2)], 'bandpass');
% 
% filtered_signal = filtfilt(b, a, pData);
% 
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
% low_cutoff = 48;
% high_cutoff = 52;
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
% % 去除异常值
% threshold = 0.000518497;
% filtered_signal_notch(abs(filtered_signal_notch) > threshold) = NaN;
% 
% % 插值处理NaN值
% filtered_signal_notch = fillmissing(filtered_signal_notch, 'linear');
% 
% % 包络线代码
% windowSize = 100;     
% [up,low] = envelope(filtered_signal_notch,windowSize,'peak');
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
% hold on;
% plot(up,'r','LineWidth',1.5); %包络线
% plot(low,'g','LineWidth',1.5); %包络线
% title('滤波后的信号');
% xlabel('时间 (s)');
% ylabel('幅度');
% legend('滤波后的信号');
% hold off;
% 
% % 设置 y 轴的范围和单位
% ylim([-4e-3 4e-3]);
% yticks([-4e-3 -3e-3 -2e-3 -1e-3 0 1e-3 2e-3 3e-3 4e-3]);
% yticklabels({'-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'});
% ylabel('幅度 (x10^{-3})');
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
% figure;
% plot(f_subset1, Pxx_subset1);
% title('带通+带阻滤波后频谱图');
% xlabel('频率 (Hz)');
% ylabel('能量');
% 



