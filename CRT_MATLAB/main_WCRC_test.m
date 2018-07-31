clear all; close all; clc
addpath(genpath('../Dataset'))
NUM = [50];

for iter = 1: length(NUM)
    % read data
    %load eight_class_Indian_image
    load indian9
    Data0 = data./max(data(:));
    [m n d] = size(Data0);
    
% load Salinas
% % Data0 = salinas./max(salinas(:));
% Data0 = salinas;
% [m n d] = size(Data0);
% Data = reshape(Data0, m*n, d);
% load Salinas_gt
% map = salinas_gt;

%  load Dataset_Image_University
% Data = DataTest./max(DataTest(:));
% % Data=DataTest;
% Data0 = reshape(Data, 340, 610, 103);
% [m n d] = size(Data0);
% load University_groundtruth_map
% map = map';

   
    % band selection
    display('band select timing:');
    tic;
    num = 10;
    ind = findstart(num, Data0);
    a = unique(ind);
    [value pos] = max(histc(ind, a));
    if a(pos) == 0
        bsn = bsl(Data0, num, num);
    else
        bsn = bsl(Data0, a(pos), num);
    end
    bandSelectTime=toc;
    display(bandSelectTime);

%     Data = reshape(Data0, m*n, d);
%     extract spatial feature
%     DataLBP2 = LBP_Feature_Extraction(Data,m*n,d);
%     Data=DataLBP2./max(DataLBP2(:));
    % finish the feature
    %load eight_class_groudtruth_map
    load indian9_map
% Data0 = reshape(DataTest, 340, 610, 103);
    r = 1;  nr = 8;
    mapping = getmapping(nr,'u2'); 
    Data = Data0(:, :, bsn(:));
    display('LBP extraction timing:');
    tic;
    Feature_P = LBP_feature_global(Data, r, nr, mapping, 10, map);
     LBP_time= toc;
     disp(LBP_time);
    d = size(Feature_P, 3);
    Data = reshape(Feature_P, m*n, d);
    
    
    no_class = max(map(:));
%     CTrain = NUM(iter) * ones(1, no_class);
    mask_train = zeros(size(map));  mask_test = mask_train;
    DataTrain = [];
    DataTest = []; CTest = [];CTrain=[];
    percentage=0.2;
    for i = 1: no_class
        tmp  = find(map==i);
        rand('seed', 1);
        index_i = randperm(length(tmp));
%         DataTrain = [DataTrain; Data(tmp(index_i(1:CTrain(i))), :)];
%         DataTest = [DataTest; Data(tmp(index_i(1:end)), :)];
%         CTest =  [CTest length(tmp(index_i(1:end)))];
        % DataTrain = [DataTrain; Data(tmp(index_i(1:ceil(length(tmp)*percentage))), :)];
        %DataTest = [DataTest; Data(tmp(index_i(1:end)), :)];
        DataTrain = [DataTrain; Data(tmp(index_i(1:100)), :)];
        DataTest = [DataTest; Data(tmp(index_i(1:(length(tmp) - 100))), :)];
        %CTest =  [CTest length(tmp(index_i(1:end)))];
        % CTrain =  [CTrain length(tmp(index_i(1:ceil(length(tmp)*percentage))))];
        CTest =  [CTest length(tmp(index_i(1:(length(tmp) - 100))))];
        CTrain =  [CTrain length(tmp(index_i(1:100)))];
    end
    Normalize = max(DataTrain(:));
    DataTrain = DataTrain./Normalize;
    DataTest = DataTest./Normalize;

    %para = [1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 5 1e1];   
    para = [1];
    
    % Residuel Fusion with Multiple Parameters
     tic;
    for index = 1: length(para)  
        output{index} = CRC_Classification_post(DataTrain, CTrain, DataTest, para(index));
        tmp = output{index};
        [value class(:, index)] = min(tmp'); 
        [confusion, accur_CRT(index), TPR, FPR] = confusion_matrix_wei(class(:, index), CTest);
        disp(accur_CRT(index));
    end
        timeTotal = toc;
        timeAve = timeTotal/ length(para);
        disp(timeAve);
        
end