clear all; close all; clc

% addpath(genpath('../Dataset'))

num_rows = 64;
num_cols = 64;
N = 169;
filename = 'hydice';


M = num_rows * num_cols;
fid = fopen([filename '.dat'], 'r', 'ieee-be');
X = fread(fid, [N M], 'float32');
fclose(fid);

fid = fopen([filename '.ground_truth.dat'], 'r', 'ieee-be');
mask = fread(fid, [1 M], 'uint8');
fclose(fid);

DataTest = reshape(X', num_rows, num_cols, N);
DataTest_ori = DataTest(:, :, :);
row = 64; col = 64; dim = 169;
A = reshape(mask, row, col);

win_out = 11; % outer window
win_in = 3; % inner window
% figure,imagesc(X');axis image
tic;
% Global RX
r1 = RX(X);
toc;
tmp = reshape(r1, 64, 64);

figure, imagesc(tmp); axis image;
title('Global RX')


%画出在门限为0.01的时候的检测图%
%disp('map a figure');
%threshold = 



disp('Running ROC...');
threshold = 0.005;
mask = reshape(mask, 1, row*col);
anomaly_map = logical(double(mask)==1);
normal_map = logical(double(mask)==0);
r_max = max(r1(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r1 > tau);
  PF1(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD1(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area1 = sum((PF1(1:end-1)-PF1(2:end)).*(PD1(2:end)+PD1(1:end-1))/2);
toc;

% Local RX
r2 = hyperRxDetector_Local(DataTest_ori, win_out, win_in);
r2 = reshape(r2, 1, M);
tmp = reshape(r2, 64, 64);
figure, imagesc(tmp); axis image;
title('Local RX')
r_max = max(r2(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r2 > tau);
  PF2(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD2(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area1 = sum((PF1(1:end-1)-PF1(2:end)).*(PD1(2:end)+PD1(1:end-1))/2);
area2 = sum((PF2(1:end-1)-PF2(2:end)).*(PD2(2:end)+PD2(1:end-1))/2);

figure,
semilogx(PF1, PD1, 'k--', 'LineWidth', 2);  hold on
semilogx(PF2, PD2, 'r--', 'LineWidth', 2);  grid on
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('Global RX','Local RX')
axis([0 0.1 0 1])