close all; clear;
% Load Data
load('D:\ops\GPclassification\CollectDatabase\Impulsive_Robustness\train_3_5_10\1sigma\training_input_data.mat');
load('D:\ops\GPclassification\CollectDatabase\Impulsive_Robustness\train_3_5_10\1sigma\training_output_data.mat');
load('D:\ops\GPclassification\CollectDatabase\Impulsive_Robustness\train_3_5_10\1sigma\testing_1_input_data.mat');
load('D:\ops\GPclassification\CollectDatabase\Impulsive_Robustness\train_3_5_10\1sigma\testing_1_output_data.mat');
load('D:\ops\GPclassification\CollectDatabase\Impulsive_Robustness\train_3_5_10\1sigma\testing_2_input_data.mat');
load('D:\ops\GPclassification\CollectDatabase\Impulsive_Robustness\train_3_5_10\1sigma\testing_2_output_data.mat');
load('D:\ops\GPclassification\CollectDatabase\Impulsive_Robustness\train_3_5_10\2sigma\testing_3_input_data.mat');
load('D:\ops\GPclassification\CollectDatabase\Impulsive_Robustness\train_3_5_10\2sigma\testing_3_output_data.mat');


% Load GPML
addpath(genpath('D:/GPRunning/gp-structure-search/gp-structure-search/source/gpml'));

%%
tic
 meanfunc = @meanZero; 
%   covfunc = @covSEiso; 
%   ell = 1.0; sf = 1.0; hyp.cov = log([ell sf]);
  covfunc = @covSEard; 
  input_size = size(training_input_data); input_size = input_size(2);
  initial_para = ones(1, input_size); sf = 1.0; cov_para = [initial_para sf];
  hyp.cov = log(cov_para);
  likfunc = @likLogistic;
%   likfunc = @likGaussian;
  infunc = @infVB;
%   infunc = @infEP;
   hyp = minimize(hyp, @gp, -150, infunc, meanfunc, covfunc,...
       likfunc, training_input_data, training_output_data);
 
   %   training: [nlZ dnlZ          ] = gp(hyp, inf, mean, cov, lik, x, y);
% prediction: [ymu ys2 fmu fs2   ] = gp(hyp, inf, mean, cov, lik, x, y, xs);
%         or: [ymu ys2 fmu fs2 lp] = gp(hyp, inf, mean, cov, lik, x, y, xs, ys);
%%
 sig_boundary = 3;
 lower_bound = -0.5;
 higher_bound = 0.5;
%% Predict the Test Data
  % Test-1
  [test_1_prediction,test_1_variance,test_1_latentmean,test_1_nlZ] =  gp(hyp, infunc, meanfunc, covfunc, likfunc, training_input_data, training_output_data, testing_1_input_data);
  % Test-2
  [test_2_prediction,test_2_variance,test_2_latentmean,test_2_nlZ] =  gp(hyp, infunc, meanfunc, covfunc, likfunc, training_input_data, training_output_data, testing_2_input_data);
  % Test-3
  [test_3_prediction,test_3_variance,test_3_latentmean,test_3_nlZ] =  gp(hyp, infunc, meanfunc, covfunc, likfunc, training_input_data, training_output_data, testing_3_input_data);

% Predicted test 1 boundary
test_1_higher_boundary = test_1_prediction+sig_boundary*sqrt(test_1_variance); 
test_1_lower_boundary = test_1_prediction-sig_boundary*sqrt(test_1_variance);

% Predicted test 2 boundary
 test_2_higher_boundary = test_2_prediction+sig_boundary*sqrt(test_2_variance); 
test_2_lower_boundary = test_2_prediction-sig_boundary*sqrt(test_2_variance);

% Predicted test 3 boundary
test_3_higher_boundary = test_3_prediction+sig_boundary*sqrt(test_3_variance); 
test_3_lower_boundary = test_3_prediction-sig_boundary*sqrt(test_3_variance);

%% Calculate the Accuracy

% test 1
index_1 = find(testing_1_output_data == 1);
index_0 = find(testing_1_output_data == -1);
test_1_output_data_1 = testing_1_output_data(index_1);
test_1_pred_1 = test_1_prediction(index_1);
test_1_output_data_0 = testing_1_output_data(index_0);
test_1_pred_0 = test_1_prediction(index_0);

num_train_1 = find(test_1_pred_0 < lower_bound);
test_1_valid_index_0 = num_train_1; % valid cases with label -1
test_1_higher_boundary_0 = test_1_higher_boundary(index_0);
test_1_lower_boundary_0 = test_1_lower_boundary(index_0);
test_1_pred_0_valid = test_1_output_data_0(num_train_1, :);
test_1_pred_0_valid_higher_boundary_H = test_1_higher_boundary_0(num_train_1);
test_1_pred_0_valid_higher_boundary_L = test_1_lower_boundary_0(num_train_1);
test_1_pred_0_valid_outBoundary_H = find(test_1_pred_0_valid > test_1_pred_0_valid_higher_boundary_H);
test_1_pred_0_valid_outBoundary_H = size(test_1_pred_0_valid_outBoundary_H); test_1_pred_0_valid_outBoundary_H = test_1_pred_0_valid_outBoundary_H(1);
test_1_pred_0_valid_outBoundary_L = find(test_1_pred_0_valid < test_1_pred_0_valid_higher_boundary_L);
test_1_pred_0_valid_outBoundary_L = size(test_1_pred_0_valid_outBoundary_L); test_1_pred_0_valid_outBoundary_L = test_1_pred_0_valid_outBoundary_L(1);
test_1_pred_0_valid_outBoundary = test_1_pred_0_valid_outBoundary_H + test_1_pred_0_valid_outBoundary_L;
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);

num_train_2 = find(test_1_pred_1 > higher_bound);
test_1_higher_boundary_1 = test_1_higher_boundary(index_1);
test_1_lower_boundary_1 = test_1_lower_boundary(index_1);
test_1_pred_1_valid = test_1_output_data_1(num_train_2, :);
test_1_pred_1_valid_higher_boundary_H = test_1_higher_boundary_1(num_train_2, :);
test_1_pred_1_valid_higher_boundary_L = test_1_lower_boundary_1(num_train_2, :);
test_1_pred_1_valid_outBoundary_H = find(test_1_pred_1_valid > test_1_pred_1_valid_higher_boundary_H);
test_1_pred_1_valid_outBoundary_H = size(test_1_pred_1_valid_outBoundary_H); test_1_pred_1_valid_outBoundary_H = test_1_pred_1_valid_outBoundary_H(1);
test_1_pred_1_valid_outBoundary_L = find(test_1_pred_1_valid < test_1_pred_1_valid_higher_boundary_L);
test_1_pred_1_valid_outBoundary_L = size(test_1_pred_1_valid_outBoundary_L); test_1_pred_1_valid_outBoundary_L = test_1_pred_1_valid_outBoundary_L(1);
test_1_pred_1_valid_outBoundary = test_1_pred_1_valid_outBoundary_H + test_1_pred_1_valid_outBoundary_L;

test_1_pred_valid_outBoundary = test_1_pred_0_valid_outBoundary + test_1_pred_1_valid_outBoundary;
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
% find invalid case outof boundary
test_1_higher_boundary_1 = test_1_higher_boundary(index_1);
test_1_higher_boundary_0 = test_1_higher_boundary(index_0);
test_1_lower_boundary_1 = test_1_lower_boundary(index_1);
test_1_lower_boundary_0 = test_1_lower_boundary(index_0);
 test_1_pred_1_outH = find(test_1_output_data_1 > test_1_higher_boundary_1);
 test_1_pred_1_outH = size(test_1_pred_1_outH); test_1_pred_1_outH = test_1_pred_1_outH(1);
 test_1_pred_1_outL = find(test_1_output_data_1 < test_1_lower_boundary_1);
 test_1_pred_1_outL = size(test_1_pred_1_outL); test_1_pred_1_outL= test_1_pred_1_outL(1);
 test_1_pred_0_outH = find(test_1_output_data_0 > test_1_higher_boundary_0);
 test_1_pred_0_outH = size(test_1_pred_0_outH); test_1_pred_0_outH = test_1_pred_0_outH(1);
 test_1_pred_0_outL = find(test_1_output_data_0 < test_1_lower_boundary_0);
 test_1_pred_0_outL = size(test_1_pred_0_outL); test_1_pred_0_outL = test_1_pred_0_outL(1);
test_1_invalid_outBoundary = test_1_pred_1_outH + test_1_pred_1_outL + test_1_pred_0_outH + test_1_pred_0_outL;
test_1_invalid_withinBoundary = test_1_size -num_valid_test_1 - test_1_invalid_outBoundary;
                                  
 Accuracy = cell(15,2);                                  
Accuracy(1:5, :) = {'Test 1 Predicted Accuracy',accuracy_test_1;
                                   'Test 1 Positive Predictive Value', test_1_PPV;
                                   'Test 1 False Discovery Rate', test_1_FDR;
                                   'Test 1 False Omission Rate', test_1_FOR;
                                   'Test 1 Negative Predictive Value', test_1_NPV;};
                                  
 Uncertainty = cell(12,2);
Uncertainty(1:4, :) = {'Test 1 Valid Prediction and within the boundary', num_valid_test_1;
                                       'Test 1 Valid Prediction but out of the boundary', test_1_pred_1_valid_outBoundary;
                                       'Test 1 Invalid Prediction but within the boundary', test_1_invalid_withinBoundary;
                                       'Test 1 Invalid Prediction but out of the boundary', test_1_invalid_outBoundary };
 %% test 2
index_1 = find(testing_2_output_data == 1);
index_0 = find(testing_2_output_data == -1);
test_2_output_data_1 = testing_2_output_data(index_1);
test_2_pred_1 = test_2_prediction(index_1);
test_2_output_data_0 = testing_2_output_data(index_0);
test_2_pred_0 = test_2_prediction(index_0);

num_train_1 = find(test_2_pred_0 < lower_bound);
test_2_valid_index_0 = num_train_1; % valid cases with label -1
test_2_higher_boundary_0 = test_2_higher_boundary(index_0);
test_2_lower_boundary_0 = test_2_lower_boundary(index_0);
test_2_pred_0_valid = test_2_output_data_0(num_train_1, :);
test_2_pred_0_valid_higher_boundary_H = test_2_higher_boundary_0(num_train_1);
test_2_pred_0_valid_higher_boundary_L = test_2_lower_boundary_0(num_train_1);
test_2_pred_0_valid_outBoundary_H = find(test_2_pred_0_valid > test_2_pred_0_valid_higher_boundary_H);
test_2_pred_0_valid_outBoundary_H = size(test_2_pred_0_valid_outBoundary_H); test_2_pred_0_valid_outBoundary_H = test_2_pred_0_valid_outBoundary_H(1);
test_2_pred_0_valid_outBoundary_L = find(test_2_pred_0_valid < test_2_pred_0_valid_higher_boundary_L);
test_2_pred_0_valid_outBoundary_L = size(test_2_pred_0_valid_outBoundary_L); test_2_pred_0_valid_outBoundary_L = test_2_pred_0_valid_outBoundary_L(1);
test_2_pred_0_valid_outBoundary = test_2_pred_0_valid_outBoundary_H + test_2_pred_0_valid_outBoundary_L;
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);

num_train_2 = find(test_2_pred_1 > higher_bound);
test_2_higher_boundary_1 = test_2_higher_boundary(index_1);
test_2_lower_boundary_1 = test_2_lower_boundary(index_1);
test_2_pred_1_valid = test_2_output_data_1(num_train_2, :);
test_2_pred_1_valid_higher_boundary_H = test_2_higher_boundary_1(num_train_2, :);
test_2_pred_1_valid_higher_boundary_L = test_2_lower_boundary_1(num_train_2, :);
test_2_pred_1_valid_outBoundary_H = find(test_2_pred_1_valid > test_2_pred_1_valid_higher_boundary_H);
test_2_pred_1_valid_outBoundary_H = size(test_2_pred_1_valid_outBoundary_H); test_2_pred_1_valid_outBoundary_H = test_2_pred_1_valid_outBoundary_H(1);
test_2_pred_1_valid_outBoundary_L = find(test_2_pred_1_valid < test_2_pred_1_valid_higher_boundary_L);
test_2_pred_1_valid_outBoundary_L = size(test_2_pred_1_valid_outBoundary_L); test_2_pred_1_valid_outBoundary_L = test_2_pred_1_valid_outBoundary_L(1);
test_2_pred_1_valid_outBoundary = test_2_pred_1_valid_outBoundary_H + test_2_pred_1_valid_outBoundary_L;

test_2_pred_valid_outBoundary = test_2_pred_0_valid_outBoundary + test_2_pred_1_valid_outBoundary;
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
% find invalid case outof boundary
test_2_higher_boundary_1 = test_2_higher_boundary(index_1);
test_2_higher_boundary_0 = test_2_higher_boundary(index_0);
test_2_lower_boundary_1 = test_2_lower_boundary(index_1);
test_2_lower_boundary_0 = test_2_lower_boundary(index_0);
 test_2_pred_1_outH = find(test_2_output_data_1 > test_2_higher_boundary_1);
 test_2_pred_1_outH = size(test_2_pred_1_outH); test_2_pred_1_outH = test_2_pred_1_outH(1);
 test_2_pred_1_outL = find(test_2_output_data_1 < test_2_lower_boundary_1);
 test_2_pred_1_outL = size(test_2_pred_1_outL); test_2_pred_1_outL= test_2_pred_1_outL(1);
 test_2_pred_0_outH = find(test_2_output_data_0 > test_2_higher_boundary_0);
 test_2_pred_0_outH = size(test_2_pred_0_outH); test_2_pred_0_outH = test_2_pred_0_outH(1);
 test_2_pred_0_outL = find(test_2_output_data_0 < test_2_lower_boundary_0);
 test_2_pred_0_outL = size(test_2_pred_0_outL); test_2_pred_0_outL = test_2_pred_0_outL(1);
test_2_invalid_outBoundary = test_2_pred_1_outH + test_2_pred_1_outL + test_2_pred_0_outH + test_2_pred_0_outL;
test_2_invalid_withinBoundary = test_2_size -num_valid_test_2 - test_2_invalid_outBoundary;

 Accuracy(6:10, :) =  {' Test 2 Predicted Accuracy',accuracy_test_2;
                                   'Test 2 Positive Predictive Value', test_2_PPV;
                                   'Test 2 False Discovery Rate', test_2_FDR;
                                   'Test 2 False Omission Rate', test_2_FOR;
                                   'Test 2 Negative Predictive Value', test_2_NPV;};
Uncertainty(5:8, :) =    {'Test 2 Valid Prediction and within the boundary', num_valid_test_2;
                                       'Test 2 Valid Prediction but out of the boundary', test_2_pred_1_valid_outBoundary;
                                       'Test 2 Invalid Prediction but within the boundary', test_2_invalid_withinBoundary;
                                       'Test 2 Invalid Prediction but out of the boundary', test_2_invalid_outBoundary };                         
                                   
%% test 3
index_1 = find(testing_3_output_data == 1);
index_0 = find(testing_3_output_data == -1);
test_3_output_data_1 = testing_3_output_data(index_1);
test_3_pred_1 = test_3_prediction(index_1);
test_3_output_data_0 = testing_3_output_data(index_0);
test_3_pred_0 = test_3_prediction(index_0);

num_train_1 = find(test_3_pred_0 < lower_bound);
test_3_valid_index_0 = num_train_1; % valid cases with label -1
test_3_higher_boundary_0 = test_3_higher_boundary(index_0);
test_3_lower_boundary_0 = test_3_lower_boundary(index_0);
test_3_pred_0_valid = test_3_output_data_0(num_train_1, :);
test_3_pred_0_valid_higher_boundary_H = test_3_higher_boundary_0(num_train_1);
test_3_pred_0_valid_higher_boundary_L = test_3_lower_boundary_0(num_train_1);
test_3_pred_0_valid_outBoundary_H = find(test_3_pred_0_valid > test_3_pred_0_valid_higher_boundary_H);
test_3_pred_0_valid_outBoundary_H = size(test_3_pred_0_valid_outBoundary_H); test_3_pred_0_valid_outBoundary_H = test_3_pred_0_valid_outBoundary_H(1);
test_3_pred_0_valid_outBoundary_L = find(test_3_pred_0_valid < test_3_pred_0_valid_higher_boundary_L);
test_3_pred_0_valid_outBoundary_L = size(test_3_pred_0_valid_outBoundary_L); test_3_pred_0_valid_outBoundary_L = test_3_pred_0_valid_outBoundary_L(1);
test_3_pred_0_valid_outBoundary = test_3_pred_0_valid_outBoundary_H + test_3_pred_0_valid_outBoundary_L;
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);

num_train_2 = find(test_3_pred_1 > higher_bound);
test_3_higher_boundary_1 = test_3_higher_boundary(index_1);
test_3_lower_boundary_1 = test_3_lower_boundary(index_1);
test_3_pred_1_valid = test_3_output_data_1(num_train_2, :);
test_3_pred_1_valid_higher_boundary_H = test_3_higher_boundary_1(num_train_2, :);
test_3_pred_1_valid_higher_boundary_L = test_3_lower_boundary_1(num_train_2, :);
test_3_pred_1_valid_outBoundary_H = find(test_3_pred_1_valid > test_3_pred_1_valid_higher_boundary_H);
test_3_pred_1_valid_outBoundary_H = size(test_3_pred_1_valid_outBoundary_H); test_3_pred_1_valid_outBoundary_H = test_3_pred_1_valid_outBoundary_H(1);
test_3_pred_1_valid_outBoundary_L = find(test_3_pred_1_valid < test_3_pred_1_valid_higher_boundary_L);
test_3_pred_1_valid_outBoundary_L = size(test_3_pred_1_valid_outBoundary_L); test_3_pred_1_valid_outBoundary_L = test_3_pred_1_valid_outBoundary_L(1);
test_3_pred_1_valid_outBoundary = test_3_pred_1_valid_outBoundary_H + test_3_pred_1_valid_outBoundary_L;

test_3_pred_valid_outBoundary = test_3_pred_0_valid_outBoundary + test_3_pred_1_valid_outBoundary;
test_3_pred_1_valid = test_3_pred_1(num_train_2, :);
num_train_2 = size(num_train_2); num_train_2 = num_train_2(1);

num_valid_test_3 = num_train_1 + num_train_2;
test_3_size = size(testing_3_output_data); test_3_size = test_3_size(1);
accuracy_test_3 = num_valid_test_3/test_3_size;
 
% confusion matrix
test_3_PPV = num_train_2 / (test_3_size/2);
test_3_FDR = (test_3_size/2 - num_train_2) / (test_3_size/2);
test_3_FOR = (test_3_size/2 - num_train_1) / (test_3_size/2);
test_3_NPV = num_train_1 / (test_3_size/2);
% find invalid case outof boundary
test_3_higher_boundary_1 = test_3_higher_boundary(index_1);
test_3_higher_boundary_0 = test_3_higher_boundary(index_0);
test_3_lower_boundary_1 = test_3_lower_boundary(index_1);
test_3_lower_boundary_0 = test_3_lower_boundary(index_0);
 test_3_pred_1_outH = find(test_3_output_data_1 > test_3_higher_boundary_1);
 test_3_pred_1_outH = size(test_3_pred_1_outH); test_3_pred_1_outH = test_3_pred_1_outH(1);
 test_3_pred_1_outL = find(test_3_output_data_1 < test_3_lower_boundary_1);
 test_3_pred_1_outL = size(test_3_pred_1_outL); test_3_pred_1_outL= test_3_pred_1_outL(1);
 test_3_pred_0_outH = find(test_3_output_data_0 > test_3_higher_boundary_0);
 test_3_pred_0_outH = size(test_3_pred_0_outH); test_3_pred_0_outH = test_3_pred_0_outH(1);
 test_3_pred_0_outL = find(test_3_output_data_0 < test_3_lower_boundary_0);
 test_3_pred_0_outL = size(test_3_pred_0_outL); test_3_pred_0_outL = test_3_pred_0_outL(1);
test_3_invalid_outBoundary = test_3_pred_1_outH + test_3_pred_1_outL + test_3_pred_0_outH + test_3_pred_0_outL;
test_3_invalid_withinBoundary = test_3_size -num_valid_test_3 - test_3_invalid_outBoundary;

 Accuracy(11:15, :) =  {'Test 3 Predicted Accuracy',accuracy_test_3;
                                   'Test 3 Positive Predictive Value', test_3_PPV;
                                   'Test 3 False Discovery Rate', test_3_FDR;
                                   'Test 3 False Omission Rate', test_3_FOR;
                                   'Test 3 Negative Predictive Value', test_3_NPV;};
Uncertainty(9:12, :) = {'Test 3 Valid Prediction and within the boundary', num_valid_test_3;
                                       'Test 3 Valid Prediction but out of the boundary', test_3_pred_1_valid_outBoundary;
                                       'Test 3 Invalid Prediction but within the boundary', test_3_invalid_withinBoundary;
                                       'Test 3 Invalid Prediction but out of the boundary', test_3_invalid_outBoundary };


% The number of invalid prediction
error_1 = find(testing_1_output_data > test_1_higher_boundary);
error_2 = find(testing_1_output_data < test_1_lower_boundary);

error_3 = find(testing_2_output_data > test_2_higher_boundary);
error_4 = find(testing_2_output_data < test_2_lower_boundary);

error_5 = find(testing_3_output_data > test_3_higher_boundary);
error_6 = find(testing_3_output_data < test_3_lower_boundary);
save('D:\ops\GPclassification\CollectDatabase\Impulsive_Robustness\train_3_5_10\2sigma\PredictedResult_2sigma.mat');

%% Plot 
% %% Test-1 result
% figure1 =  figure('WindowState','maximized');
% plot(test_1_prediction, '-*', 'MarkerSize', 12,'LineWidth',2);
% hold on
% plot(testing_1_output_data, '-*', 'MarkerSize', 12,'LineWidth',2);
% plot(test_1_higher_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
% plot(test_1_lower_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
% legend('Pred', 'Truth', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
% set(gca,'FontSize',18, 'FontWeight', 'bold');
% % Test-2 result
% figure2 =  figure('WindowState','maximized');
% plot(test_2_prediction, '-*', 'MarkerSize', 12,'LineWidth',2);
% hold on
% plot(testing_2_output_data, '-*', 'MarkerSize', 12,'LineWidth',2);
% plot(test_2_higher_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
% plot(test_2_lower_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
% legend('Pred', 'Truth', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
% set(gca,'FontSize',18, 'FontWeight', 'bold');

%% Test-3 result
figure3 =  figure('WindowState','maximized');
plot(test_3_prediction, '-*', 'MarkerSize', 12,'LineWidth',2);
hold on
plot(testing_3_output_data, '-*', 'MarkerSize', 12,'LineWidth',2);
plot(test_3_higher_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
plot(test_3_lower_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
legend('Pred', 'Truth', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
set(gca,'FontSize',18, 'FontWeight', 'bold');

%% Detail Plot of Test-3
plot_index = error_6(3);
plot_range = [plot_index-5:plot_index+5];
figure4 =  figure('WindowState','maximized');
plot(test_3_prediction(plot_range, :), '-*', 'MarkerSize', 12,'LineWidth',2);
hold on
plot(testing_3_output_data(plot_range, :), '-*', 'MarkerSize', 12,'LineWidth',2);
plot(test_3_higher_boundary(plot_range, :), '--o', 'MarkerSize', 12,'LineWidth',2);
plot(test_3_lower_boundary(plot_range, :), '--o', 'MarkerSize', 12,'LineWidth',2);
legend('Pred', 'Truth', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
set(gca,'FontSize',18, 'FontWeight', 'bold');
title('Test-3: Valid Case and Invalid Case but Out of Boundary ', 'FontSize',24, 'FontWeight', 'bold');

plot_index = 87709;
plot_range = [plot_index-5:plot_index+5];
figure4 =  figure('WindowState','maximized');
plot(test_3_prediction(plot_range, :), '-*', 'MarkerSize', 12,'LineWidth',2);
hold on
plot(testing_3_output_data(plot_range, :), '-*', 'MarkerSize', 12,'LineWidth',2);
plot(test_3_higher_boundary(plot_range, :), '--o', 'MarkerSize', 12,'LineWidth',2);
plot(test_3_lower_boundary(plot_range, :), '--o', 'MarkerSize', 12,'LineWidth',2);
legend('Pred', 'Truth', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
set(gca,'FontSize',18, 'FontWeight', 'bold');
title('Test-3: Valid Case and Invalid Case but Within Boundary ', 'FontSize',24, 'FontWeight', 'bold');
