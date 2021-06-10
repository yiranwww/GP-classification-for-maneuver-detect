%% Autoencoder-Classification Result
clear; clc;
% Load the original output
load('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_1_output_data.mat');
load('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_2_output_data.mat');
load('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_3_output_data.mat');
% Load the predicted result
load('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\AutoencoderResult\test_1_prediction.mat');
test_1_prediction = double(test_1_prediction);
load('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\AutoencoderResult\test_2_prediction.mat');
test_2_prediction = double(test_2_prediction);
load('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\AutoencoderResult\test_3_prediction.mat');
test_3_prediction = double(test_3_prediction);

%% Calculate the accuracy
% Judgement boundary
 lower_bound = 0.1;
 higher_bound = 0.9;
 
 % test 1
index_1 = find(testing_1_output_data == 1);
index_0 = find(testing_1_output_data == 0);
test_1_output_data_1 = testing_1_output_data(index_1);
test_1_pred_1 = test_1_prediction(index_1);
test_1_output_data_0 = testing_1_output_data(index_0);
test_1_pred_0 = test_1_prediction(index_0);

num_train_1 = find(test_1_pred_0 < lower_bound);
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);
num_train_2 = find(test_1_pred_1 > higher_bound);
num_train_2 = size(num_train_2); num_train_2 = num_train_2(1);
num_valid_test_1 = num_train_1 + num_train_2;
test_1_size = size(testing_1_output_data); test_1_size = test_1_size(1);
accuracy_test_1 = num_valid_test_1/test_1_size;

% test 2
index_1 = find(testing_2_output_data == 1);
index_0 = find(testing_2_output_data == 0);
test_2_output_data_1 = testing_2_output_data(index_1);
test_2_pred_1 = test_2_prediction(index_1);
test_2_output_data_0 = testing_2_output_data(index_0);
test_2_pred_0 = test_2_prediction(index_0);

num_train_1 = find(test_2_pred_0 < lower_bound);
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);
num_train_2 = find(test_2_pred_1 > higher_bound);
num_train_2 = size(num_train_2); num_train_2 = num_train_2(1);
num_valid_test_2 = num_train_1 + num_train_2;
test_2_size = size(testing_2_output_data); test_2_size = test_2_size(1);
accuracy_test_2 = num_valid_test_2/test_2_size;

% test 3
index_1 = find(testing_3_output_data == 1);
index_0 = find(testing_3_output_data == 0);
test_3_output_data_1 = testing_3_output_data(index_1);
test_3_pred_1 = test_3_prediction(index_1);
test_3_output_data_0 = testing_3_output_data(index_0);
test_3_pred_0 = test_3_prediction(index_0);

num_train_1 = find(test_3_pred_0 < lower_bound);
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);
num_train_2 = find(test_3_pred_1 > higher_bound);
num_train_2 = size(num_train_2); num_train_2 = num_train_2(1);
num_valid_test_3 = num_train_1 + num_train_2;
test_3_size = size(testing_3_output_data); test_3_size = test_3_size(1);
accuracy_test_3 = num_valid_test_3/test_3_size;

save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\AutoencoderResult\AccuracyResult.mat');