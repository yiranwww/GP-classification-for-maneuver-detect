%% Using PCA  to do the classification
load('D:\ops\GPclassification\CollectDatabase\All_Robustness\1sigma\training_input_data.mat');
load('D:\ops\GPclassification\CollectDatabase\All_Robustness\1sigma\training_output_data.mat');
load('D:\ops\GPclassification\CollectDatabase\All_Robustness\1sigma\testing_1_input_data.mat');
load('D:\ops\GPclassification\CollectDatabase\All_Robustness\1sigma\testing_1_output_data.mat');
load('D:\ops\GPclassification\CollectDatabase\All_Robustness\1sigma\testing_2_input_data.mat');
load('D:\ops\GPclassification\CollectDatabase\All_Robustness\1sigma\testing_2_output_data.mat');
load('D:\ops\GPclassification\CollectDatabase\All_Robustness\1sigma\testing_3_input_data.mat');
load('D:\ops\GPclassification\CollectDatabase\All_Robustness\1sigma\testing_3_output_data.mat');

% pca test
training_sample =  [training_input_data training_output_data];
rng(4)
[coeff,scoreTrain,latent,tsquared,explained,mu] = pca(training_input_data);
sum_explained = 0;
idx = 0;
while sum_explained < 95
    idx = idx + 1;
    sum_explained = sum_explained + explained(idx);
end
idx

scoreTrain95 = scoreTrain(:, 1:idx);
mdl = fitctree(scoreTrain95,training_output_data);

scoreTest_1 = (testing_1_input_data-mu)*coeff(:,1:idx);
scoreTest_2 = (testing_2_input_data-mu)*coeff(:,1:idx);
scoreTest_3 = (testing_3_input_data-mu)*coeff(:,1:idx);

test_1_prediction = predict(mdl, scoreTest_1);
test_2_prediction = predict(mdl, scoreTest_2);
test_3_prediction = predict(mdl, scoreTest_3);

% Accuracy 
 lower_bound = -0.5;
 higher_bound = 0.5;
 
 % Test-1
 % test 1
index_1 = find(testing_1_output_data == 1);
index_0 = find(testing_1_output_data == -1);
test_1_output_data_1 = testing_1_output_data(index_1);
test_1_pred_1 = test_1_prediction(index_1);
test_1_output_data_0 = testing_1_output_data(index_0);
test_1_pred_0 = test_1_prediction(index_0);

num_train_1 = find(test_1_pred_0 < lower_bound);
test_1_valid_index_0 = num_train_1; % valid cases with label -1
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);

num_train_2 = find(test_1_pred_1 > higher_bound);

test_1_pred_1_valid = test_1_output_data_1(num_train_2, :);
test_1_pred_1_valid = test_1_pred_1(num_train_2, :);
num_train_2 = size(num_train_2); num_train_2 = num_train_2(1);

num_valid_test_1 = num_train_1 + num_train_2;
test_1_size = size(testing_1_output_data); test_1_size = test_1_size(1);
accuracy_test_1 = num_valid_test_1/test_1_size;
 
% confusion matrix
test_1_PPV = num_train_2 / (test_1_size/2);
test_1_FDR = (test_1_size/2 - num_train_2) / (test_1_size/2);
test_1_FOR = (test_1_size/2 - num_train_1) / (test_1_size/2);
test_1_NPV = num_train_1 / (test_1_size/2);

 % test 2
index_1 = find(testing_2_output_data == 1);
index_0 = find(testing_2_output_data == -1);
test_2_output_data_1 = testing_2_output_data(index_1);
test_2_pred_1 = test_2_prediction(index_1);
test_2_output_data_0 = testing_2_output_data(index_0);
test_2_pred_0 = test_2_prediction(index_0);

num_train_1 = find(test_2_pred_0 < lower_bound);
test_2_valid_index_0 = num_train_1; % valid cases with label -1
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);

num_train_2 = find(test_2_pred_1 > higher_bound);

test_2_pred_1_valid = test_2_output_data_1(num_train_2, :);
test_2_pred_1_valid = test_2_pred_1(num_train_2, :);
num_train_2 = size(num_train_2); num_train_2 = num_train_2(1);

num_valid_test_2 = num_train_1 + num_train_2;
test_2_size = size(testing_2_output_data); test_2_size = test_2_size(1);
accuracy_test_2 = num_valid_test_2/test_2_size;
 
% confusion matrix
test_2_PPV = num_train_2 / (test_2_size/2);
test_2_FDR = (test_2_size/2 - num_train_2) / (test_2_size/2);
test_2_FOR = (test_2_size/2 - num_train_1) / (test_2_size/2);
test_2_NPV = num_train_1 / (test_2_size/2);

 % test 3
index_1 = find(testing_3_output_data == 1);
index_0 = find(testing_3_output_data == -1);
test_3_output_data_1 = testing_3_output_data(index_1);
test_3_pred_1 = test_3_prediction(index_1);
test_3_output_data_0 = testing_3_output_data(index_0);
test_3_pred_0 = test_3_prediction(index_0);

num_train_1 = find(test_3_pred_0 < lower_bound);
test_3_valid_index_0 = num_train_1; % valid cases with label -1
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);

num_train_2 = find(test_3_pred_1 > higher_bound);

test_3_pred_1_valid = test_3_output_data_1(num_train_2, :);
test_3_pred_1_valid = test_3_pred_1(num_train_2, :);
num_train_2 = size(num_train_2); num_train_2 = num_train_2(1);

num_valid_test_3 = num_train_1 + num_train_2;
test_3_size = size(testing_3_output_data); test_3_size = test_3_size(1);
digits(8); %定义精度
accuracy_test_3 = vpa(num_valid_test_3/test_3_size)
 
% confusion matrix
test_3_PPV = num_train_2 / (test_3_size/2);
test_3_FDR = (test_3_size/2 - num_train_2) / (test_3_size/2);
test_3_FOR = (test_3_size/2 - num_train_1) / (test_3_size/2);
test_3_NPV = num_train_1 / (test_3_size/2);
