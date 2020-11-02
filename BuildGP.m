% close all; clear;
% Load Data
load('D:\ops\GPclassification\Final_code\DataCollectionFile\GP_test\training_input_data.mat');
load('D:\ops\GPclassification\Final_code\DataCollectionFile\GP_test\training_output_data.mat');
load('D:\ops\GPclassification\Final_code\DataCollectionFile\GP_test\testing_1_input_data.mat');
load('D:\ops\GPclassification\Final_code\DataCollectionFile\GP_test\testing_1_output_data.mat');
load('D:\ops\GPclassification\Final_code\DataCollectionFile\GP_test\testing_2_input_data.mat');
load('D:\ops\GPclassification\Final_code\DataCollectionFile\GP_test\testing_2_output_data.mat');
load('D:\ops\GPclassification\Final_code\DataCollectionFile\GP_test\testing_3_input_data.mat');
load('D:\ops\GPclassification\Final_code\DataCollectionFile\GP_test\testing_3_output_data.mat');


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
  [a b c d lp] = gp(hyp, infunc, meanfunc, covfunc, likfunc, training_input_data, training_output_data, testing_2_input_data, testing_2_output_data);
  [train_prediction,train_means,train_variances,train_nlZ] =  gp(hyp, infunc, meanfunc, covfunc, likfunc, training_input_data, training_output_data, training_input_data);
[test_1_prediction,test_1_means,test_1_variances,test_1_nlZ] =  gp(hyp, infunc, meanfunc, covfunc, likfunc, training_input_data, training_output_data, testing_1_input_data);
  [test_2_prediction,test_2_means,test_2_variances,test_2_nlZ] =  gp(hyp, infunc, meanfunc, covfunc, likfunc, training_input_data, training_output_data, testing_2_input_data);
[test_3_prediction,test_3_means,test_3_variances,test_3_nlZ] =  gp(hyp, infunc, meanfunc, covfunc, likfunc, training_input_data, training_output_data, testing_3_input_data);
test_2_probobility = exp(lp);

toc % calculate running time
%% predicted test 1 boundary
 test_1_higher_boundary = test_1_prediction+2*sqrt(test_1_means); 
test_1_lower_boundary = test_1_prediction-2*sqrt(test_1_means);

% predicted test 2 boundary
 test_2_higher_boundary = test_2_prediction+2*sqrt(test_2_means); 
test_2_lower_boundary = test_2_prediction-2*sqrt(test_2_means);

% predicted test 3 boundary
 [test_3_prediction, sig2_3] =  gp(hyp, @infVB, meanfunc, covfunc, likfunc, training_input_data, training_output_data, testing_3_input_data);
test_3_higher_boundary = test_3_prediction+2*sqrt(sig2_3); 
test_3_lower_boundary = test_3_prediction-2*sqrt(sig2_3);

%%
 lower_bound = -0.5;
 higher_bound = 0.5;
% test 1
index_1 = []; index_0 = [];
 index_1 = find(testing_1_output_data == 1);
index_0 = find(testing_1_output_data == -1);
test_1_output_data_1 = testing_1_output_data(index_1);
test_1_pred_1 = test_1_prediction(index_1);
test_1_output_data_0 = testing_1_output_data(index_0);
test_1_pred_0 = test_1_prediction(index_0);
% out of boundaries
test_1_higher_boundary_1 = test_1_higher_boundary(index_1);
test_1_true_1_out = find(test_1_output_data_1 >test_1_higher_boundary_1);
test_1_higher_boundary_0 = test_1_higher_boundary(index_0);
test_1_true_0_out = find(test_1_output_data_0 >test_1_higher_boundary_0);

num_train_1 = find(test_1_pred_0 < lower_bound);
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);
num_train_2 = find(test_1_pred_1 > higher_bound);
test_1_index_right_1 = num_train_2;
num_train_2 = size(num_train_2); num_train_2 = num_train_2(1);
num_valid_test_1 = num_train_1 + num_train_2;
test_1_size = size(testing_1_output_data); test_1_size = test_1_size(1);
accuracy_test_1 = num_valid_test_1/test_1_size

test_1_higher_boundary_1 = test_1_higher_boundary(index_1);
test_1_higher_boundary_0 = test_1_higher_boundary(index_0);
test_1_lower_boundary_1 = test_1_lower_boundary(index_1);
test_1_lower_boundary_0 = test_1_lower_boundary(index_0);

 %% test 2
 index_1 = find(testing_2_output_data == 1);
index_0 = find(testing_2_output_data == -1);
output_data_1 = testing_2_output_data(index_1);
test_pred_1 = test_2_prediction(index_1);
output_data_0 = testing_2_output_data(index_0);
test_pred_0 = test_2_prediction(index_0);

test_higher_boundary_1 = test_2_higher_boundary(index_1);
test_higher_boundary_0 = test_2_higher_boundary(index_0);
test_lower_boundary_1 = test_2_lower_boundary(index_1);
test_lower_boundary_0 = test_2_lower_boundary(index_0);

num_test_0 = find(test_pred_0 < lower_bound);
num_test_0 = size(num_test_0); num_test_0 = num_test_0(1);
num_test_1 = find(test_pred_1 > higher_bound);
num_test_1 = size(num_test_1); num_test_1 = num_test_1(1);
num_valid_test_2 = num_test_0 + num_test_1;
test_2_size = size(testing_2_output_data); test_2_size = test_2_size(1);
accuracy_test_2 = num_valid_test_2/test_2_size

test_2_output_data_0 = testing_2_output_data(index_0);
test_2_output_data_1 = testing_2_output_data(index_1);
test_2_higher_boundary_1 = test_2_higher_boundary(index_1);
test_2_true_1_out = find(test_2_output_data_1 >test_2_higher_boundary_1);
test_2_higher_boundary_0 = test_2_higher_boundary(index_0);
test_2_true_0_out = find(test_2_output_data_0 >test_2_higher_boundary_0);
%% test 3

index_1 = []; index_0 = [];
 index_1 = find(testing_3_output_data == 1);
index_0 = find(testing_3_output_data == -1);
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
accuracy_test_3 = num_valid_test_3/test_3_size

%  training
index_1 = []; index_0 = [];
 index_1 = find(training_output_data == 1);
index_0 = find(training_output_data == -1);
train_output_data_1 = training_output_data(index_1);
train_pred_1 = train_prediction(index_1);
train_output_data_0 = training_output_data(index_0);
train_pred_0 = train_prediction(index_0);

num_part_1 = find(train_pred_0 < lower_bound);
num_part_1 = size(num_part_1); num_part_1 = num_part_1(1);
num_part_2 = find(train_pred_1 > higher_bound);
num_part_2 = size(num_part_2); num_part_2 = num_part_2(1);
num_part = num_part_1 + num_part_2;
part_size = size(training_output_data); part_size = part_size(1);
accuracy_train = num_part/part_size

save('D:\ops\GPclassification\Final_code\DataCollectionFile\GP_test\GPmodel.mat');
%%
figure1 =  figure('WindowState','maximized');
subplot(3,1,1)
 plot(test_2_prediction(1:20,1), '--o', 'MarkerSize', 12,'LineWidth',2)
hold on
plot(testing_2_output_data(1:20,1), '--*', 'MarkerSize', 12,'LineWidth',2)
set(gca,'FontSize',18, 'FontWeight', 'bold');
legend( 'Pred', 'Truth', 'FontSize', 14);
ylabel('Output','FontSize', 24, 'FontWeight', 'bold')
subplot(3,1,2)
stem(test_2_probobility(1:20,1), '--', 'MarkerSize', 12,'LineWidth',2)
set(gca,'FontSize',18, 'FontWeight', 'bold');
ylabel('Probability','FontSize', 24, 'FontWeight', 'bold')
subplot(3,1,3)
error = testing_2_output_data - test_2_prediction;
stem(abs(test_2_prediction(1:20,1)), 'r-', 'MarkerSize', 12,'LineWidth',2);
hold on
stem(test_2_probobility(1:20,1), 'b--', 'MarkerSize', 12,'LineWidth',2)

%%
figure2 =  figure('WindowState','maximized');
plot(test_2_prediction, '-*', 'MarkerSize', 12,'LineWidth',2);
hold on
plot(testing_2_output_data, '-*', 'MarkerSize', 12,'LineWidth',2);
plot(test_2_higher_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
plot(test_2_lower_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
legend('Pred', 'Truth', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
set(gca,'FontSize',18, 'FontWeight', 'bold');
%%
error_1 = find(testing_1_output_data > test_1_higher_boundary);
error_2 = find(testing_1_output_data < test_1_lower_boundary);

error_3 = find(testing_2_output_data > test_2_higher_boundary);
error_4 = find(testing_2_output_data < test_2_lower_boundary);

error_5 = find(testing_3_output_data > test_3_higher_boundary);
error_6 = find(testing_3_output_data < test_3_lower_boundary);


%% Invalid cases plot Test Data
invalid_test_index_0 = find(test_pred_0 > lower_bound);
invalid_test_index_1 = find(test_pred_1 < higher_bound);
% corresponding result
invalid_label_0 = output_data_0(invalid_test_index_0);
invalid_label_1 = output_data_1(invalid_test_index_1);

invalid_test_out_1 = test_pred_1(invalid_test_index_1);
invalid_test_out_0 = test_pred_0(invalid_test_index_0);

invalid_higher_boundary_1  = test_higher_boundary_1(invalid_test_index_1);
invalid_higher_boundary_0  = test_higher_boundary_0(invalid_test_index_0);
invalid_lower_boundary_1 = test_lower_boundary_1(invalid_test_index_1);
invalid_lower_boundary_0 = test_lower_boundary_0(invalid_test_index_0);

%plot
figure3 =  figure('WindowState','maximized');
plot(invalid_label_1, '-*', 'MarkerSize', 12,'LineWidth',2);
hold on
plot(invalid_test_out_1, '-*', 'MarkerSize', 12,'LineWidth',2);
plot(invalid_higher_boundary_1, '--o', 'MarkerSize', 12,'LineWidth',2);
plot(invalid_lower_boundary_1, '--o', 'MarkerSize', 12,'LineWidth',2);
legend('Truth', 'Pred', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
set(gca,'FontSize',18, 'FontWeight', 'bold');
title('Label 1','FontSize',26, 'FontWeight', 'bold' )

figure4 =  figure('WindowState','maximized');
plot(invalid_label_0, '-*', 'MarkerSize', 12,'LineWidth',2);
hold on
plot(invalid_test_out_0, '-*', 'MarkerSize', 12,'LineWidth',2);
plot(invalid_higher_boundary_0, '--o', 'MarkerSize', 12,'LineWidth',2);
plot(invalid_lower_boundary_0, '--o', 'MarkerSize', 12,'LineWidth',2);
legend('Truth', 'Pred', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
set(gca,'FontSize',18, 'FontWeight', 'bold');
title('Label -1','FontSize',26, 'FontWeight', 'bold' )
%% Hyperparameter
figure5 =  figure('WindowState','maximized');
cov_para = hyp.cov;
stem(cov_para, 'b-', 'MarkerSize', 10,'LineWidth',2);
set(gca,'FontSize',18, 'FontWeight', 'bold');
title('Group 5: Hyperparameter','FontSize',26, 'FontWeight', 'bold' )

%% test-3 result
figure6 =  figure('WindowState','maximized');
plot(test_3_prediction, '-*', 'MarkerSize', 12,'LineWidth',2);
hold on
plot(testing_3_output_data, '-*', 'MarkerSize', 12,'LineWidth',2);
plot(test_3_higher_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
plot(test_3_lower_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
legend('Pred', 'Truth', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
set(gca,'FontSize',18, 'FontWeight', 'bold');
