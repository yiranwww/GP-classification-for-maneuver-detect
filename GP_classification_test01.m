close all; clear;
% Load Data
load('D:\ops\GPclassification\CollectDatabase\small_maneuver\runningstep150\results-20200624-10mps\train_input.mat');
load('D:\ops\GPclassification\CollectDatabase\small_maneuver\runningstep150\results-20200624-10mps\train_output.mat');
load('D:\ops\GPclassification\CollectDatabase\small_maneuver\runningstep150\results-20200624-10mps\test_input_total.mat');
load('D:\ops\GPclassification\CollectDatabase\small_maneuver\runningstep150\results-20200624-10mps\test_output_data.mat');
load('D:\ops\GPclassification\CollectDatabase\small_maneuver\runningstep150\results-20200624-10mps\train_input_total.mat');
load('D:\ops\GPclassification\CollectDatabase\small_maneuver\runningstep150\results-20200624-10mps\output_data.mat');

% Load GPML
addpath(genpath('D:/GPRunning/gp-structure-search/gp-structure-search/source/gpml'));

%%
tic
 meanfunc = @meanZero; 
%   covfunc = @covSEiso; 
%   ell = 1.0; sf = 1.0; hyp.cov = log([ell sf]);
  covfunc = @covSEard; 
  input_size = size(train_input); input_size = input_size(2);
  initial_para = ones(1, input_size); sf = 1.0; cov_para = [initial_para sf];
  hyp.cov = log(cov_para);
  likfunc = @likLogistic;
%   likfunc = @likGaussian;
  infunc = @infVB;
%   infunc = @infEP;
   hyp = minimize(hyp, @gp, -20, infunc, meanfunc, covfunc,...
       likfunc, train_input, train_output);
 
   %   training: [nlZ dnlZ          ] = gp(hyp, inf, mean, cov, lik, x, y);
% prediction: [ymu ys2 fmu fs2   ] = gp(hyp, inf, mean, cov, lik, x, y, xs);
%         or: [ymu ys2 fmu fs2 lp] = gp(hyp, inf, mean, cov, lik, x, y, xs, ys);
  [a b c d lp] = gp(hyp, infunc, meanfunc, covfunc, likfunc, train_input, train_output, test_input_total, test_output_data);
[test_prediction,test_means,test_variances,test_nlZ] =  gp(hyp, infunc, meanfunc, covfunc, likfunc, train_input, train_output, test_input_total);
[train_prediction,train_means,train_variances,train_nlZ] =  gp(hyp, infunc, meanfunc, covfunc, likfunc, train_input, train_output, train_input_total);
[part_prediction,part_means,part_variances,part_nlZ] =  gp(hyp, infunc, meanfunc, covfunc, likfunc, train_input, train_output, train_input);  
% [p,mu,s2,nlZ] =  binaryGP(hyp, @infEP, covfunc, likfunc, train_input, train_output, test_input_total);
%   [me, sig2] =  gp(hyp, @infVB, meanfunc, covfunc, likfunc, train_input, train_output, test_input_total);
%% predicted boundary
 [me, sig2] =  gp(hyp, @infVB, meanfunc, covfunc, likfunc, train_input, train_output, test_input_total);
test_higher_boundary = me+2*sqrt(sig2); 
test_lower_boundary = me-2*sqrt(sig2);
% test_likelihood = exp(test_nlZ);

test_probobility = exp(lp);
%%
 lower_bound = -0.5;
 higher_bound = 0.5;
 % test
 index_1 = find(test_output_data == 1);
index_0 = find(test_output_data == -1);
output_data_1 = test_output_data(index_1);
test_pred_1 = test_prediction(index_1);
output_data_0 = test_output_data(index_0);
test_pred_0 = test_prediction(index_0);

test_higher_boundary_1 = test_higher_boundary(index_1);
test_higher_boundary_0 = test_higher_boundary(index_0);
test_lower_boundary_1 = test_lower_boundary(index_1);
test_lower_boundary_0 = test_lower_boundary(index_0);

num_test_1 = find(test_pred_0 < lower_bound);
num_test_1 = size(num_test_1); num_test_1 = num_test_1(1);
num_test_2 = find(test_pred_1 > higher_bound);
num_test_2 = size(num_test_2); num_test_2 = num_test_2(1);
num_test = num_test_1 + num_test_2;
test_size = size(test_output_data); test_size = test_size(1);
accuracy_test = num_test/test_size

% train
index_1 = []; index_0 = [];
 index_1 = find(output_data == 1);
index_0 = find(output_data == -1);
train_output_data_1 = output_data(index_1);
train_pred_1 = train_prediction(index_1);
train_output_data_0 = output_data(index_0);
train_pred_0 = train_prediction(index_0);

num_train_1 = find(train_pred_0 < lower_bound);
num_train_1 = size(num_train_1); num_train_1 = num_train_1(1);
num_train_2 = find(train_pred_1 > higher_bound);
num_train_2 = size(num_train_2); num_train_2 = num_train_2(1);
num_train = num_train_1 + num_train_2;
train_size = size(output_data); train_size = train_size(1);
accuracy_train = num_train/train_size

% part training
index_1 = []; index_0 = [];
 index_1 = find(train_output == 1);
index_0 = find(train_output == -1);
part_output_data_1 = train_output(index_1);
part_pred_1 = part_prediction(index_1);
part_output_data_0 = train_output(index_0);
part_pred_0 = part_prediction(index_0);

num_part_1 = find(part_pred_0 < lower_bound);
num_part_1 = size(num_part_1); num_part_1 = num_part_1(1);
num_part_2 = find(part_pred_1 > higher_bound);
num_part_2 = size(num_part_2); num_part_2 = num_part_2(1);
num_part = num_part_1 + num_part_2;
part_size = size(train_output); part_size = part_size(1);
accuracy_part = num_part/part_size
save('D:\ops\GPclassification\CollectDatabase\small_maneuver\runningstep150\results-20200624-10mps\PredictedResult.mat');
tic
%%
% figure1 =  figure('WindowState','maximized');
% subplot(3,1,1)
%  plot(test_prediction(1:20,1), '--o', 'MarkerSize', 12,'LineWidth',2)
% hold on
% plot(test_output_data(1:20,1), '--*', 'MarkerSize', 12,'LineWidth',2)
% set(gca,'FontSize',18, 'FontWeight', 'bold');
% legend( 'Pred', 'Truth', 'FontSize', 14);
% ylabel('Output','FontSize', 24, 'FontWeight', 'bold')
% subplot(3,1,2)
% stem(test_probobility(1:20,1), '--', 'MarkerSize', 12,'LineWidth',2)
% set(gca,'FontSize',18, 'FontWeight', 'bold');
% ylabel('Probability','FontSize', 24, 'FontWeight', 'bold')
% subplot(3,1,3)
% error = test_output_data - test_prediction;
% stem(abs(test_prediction(1:20,1)), 'r-', 'MarkerSize', 12,'LineWidth',2);
% hold on
% stem(test_probobility(1:20,1), 'b--', 'MarkerSize', 12,'LineWidth',2)
% 
% %%
% figure2 =  figure('WindowState','maximized');
% plot(test_prediction, '-*', 'MarkerSize', 12,'LineWidth',2);
% hold on
% plot(test_output_data, '-*', 'MarkerSize', 12,'LineWidth',2);
% plot(test_higher_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
% plot(test_lower_boundary, '--o', 'MarkerSize', 12,'LineWidth',2);
% legend('Pred', 'Truth', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
% set(gca,'FontSize',18, 'FontWeight', 'bold');

error_1 = find(test_output_data > test_higher_boundary);
error_2 = find(test_output_data < test_lower_boundary);


%% total lp verify
LP = exp(lp);
n = size(LP); n = n(1);
for i = 1:n
    lp_verify(i) = abs(LP(i) * test_output_data(i) + (1 - LP(i)) * (-test_output_data(i)));
end
lp_verify = lp_verify';


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
% figure3 =  figure('WindowState','maximized');
% plot(invalid_label_1, '-*', 'MarkerSize', 12,'LineWidth',2);
% hold on
% plot(invalid_test_out_1, '-*', 'MarkerSize', 12,'LineWidth',2);
% plot(invalid_higher_boundary_1, '--o', 'MarkerSize', 12,'LineWidth',2);
% plot(invalid_lower_boundary_1, '--o', 'MarkerSize', 12,'LineWidth',2);
% legend('Truth', 'Pred', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
% set(gca,'FontSize',18, 'FontWeight', 'bold');
% title('Label 1','FontSize',26, 'FontWeight', 'bold' )
% 
% figure4 =  figure('WindowState','maximized');
% plot(invalid_label_0, '-*', 'MarkerSize', 12,'LineWidth',2);
% hold on
% plot(invalid_test_out_0, '-*', 'MarkerSize', 12,'LineWidth',2);
% plot(invalid_higher_boundary_0, '--o', 'MarkerSize', 12,'LineWidth',2);
% plot(invalid_lower_boundary_0, '--o', 'MarkerSize', 12,'LineWidth',2);
% legend('Truth', 'Pred', 'Higher Boundary', 'Lower Boundary','FontSize', 14);
% set(gca,'FontSize',18, 'FontWeight', 'bold');
% title('Label -1','FontSize',26, 'FontWeight', 'bold' )

