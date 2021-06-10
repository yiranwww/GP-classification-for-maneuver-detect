%% AutoEncoder Mean Accuracy Value
clear;
seed_num = [1:10];
test_1 = []; test_2 = []; test_3 = [];
for i = seed_num
file_name = ['D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\AutoencoderResult\seed', num2str(i), '\AccuracyResult.mat'];
load(file_name, 'accuracy_test_1', 'accuracy_test_2', 'accuracy_test_3');
test_1 = [test_1, accuracy_test_1];
test_2 = [test_2, accuracy_test_2];
test_3 = [test_3, accuracy_test_3];
end
% mean value
digits(8); %定义精度
mean_test_1 = vpa(mean(test_1));
mean_test_2 = vpa(mean(test_2));
mean_test_3 = vpa(mean(test_3))