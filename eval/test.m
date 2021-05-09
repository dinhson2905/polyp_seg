close all;
clc;

Thresholds = 1:-1/255:0;
ResultMapPath = '../results/HarDCPD/CVC-300/';
gtPath = '../data/TestDataset/CVC-300/masks/';

imgFile = dir([ResultMapPath '149.png']);
name = imgFile.name;
gt = imread([gtPath name]);
if (ndims(gt) > 2)
    gt = rgb2gray(gt);
end

if ~islogical(gt)
    gt = gt(:,:,1) > 128;
end

resmap = imread([ResultMapPath name]);
resmap = im2double(resmap(:,:,1));
resmap = reshape(mapminmax(resmap(:)',0,1),size(resmap));

[Smeasure, wFmeasure, MAE] = deal(0);
Smeasure = StructureMeasure(resmap, logical(gt));
wFmeasure = original_WFb(resmap, logical(gt));
MAE = mean2(abs(double(logical(gt)) - resmap));

[threshold_Fmeasure, threshold_Emeasure, threshold_IoU] = deal(zeros(1, length(Thresholds)));
[threshold_Precion, threshold_Recall] = deal(zeros(1, length(Thresholds)));
[threshold_Sensitivity, threshold_Specificity, threshold_Dice] = deal(zeros(1, length(Thresholds)));

[threshold_E, threshold_F, threshold_Pr, threshold_Rec, threshold_Iou] = deal(zeros(1, length(Thresholds)));
[threshold_Spe, threshold_Dic]  = deal(zeros(1,length(Thresholds)));
for t = 1:length(Thresholds)
    threshold = Thresholds(t);
    [threshold_Pr(t), threshold_Rec(t), threshold_Spe(t), threshold_Dic(t), threshold_F(t), threshold_Iou(t)] = Fmeasure_calu(resmap, double(gt), size(gt), threshold);

    Bi_resmap = zeros(size(resmap));
    Bi_resmap(resmap>=threshold)=1;
    threshold_E(t) = Enhancedmeasure(Bi_resmap, gt);
end

threshold_Emeasure(:) = threshold_E;
threshold_Fmeasure(:) = threshold_F;
threshold_Sensitivity(:) = threshold_Rec;
threshold_Specificity(:) = threshold_Spe;
threshold_Dice(:) = threshold_Dic;
threshold_IoU(:) = threshold_Iou;

column_Dic = mean(threshold_Dice, 1);
meanDic = mean(column_Dic);