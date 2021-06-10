%% Based on Test-03
% To expolore the relationship between the uncertainty and the random seed
clear; clc;
random_seed = [2 3 4 5 7 8 10 11 12];
num_of_seed = size(random_seed); num_of_seed = num_of_seed(2);
Test_1_Invalid_Within_Percentage = []; Test_1_Invalid_Outof_Percentage = [];
Test_2_Invalid_Within_Percentage = []; Test_2_Invalid_Outof_Percentage = [];
Test_3_Invalid_Within_Percentage = []; Test_3_Invalid_Outof_Percentage = [];
Test_1_Valid_Within_Percentage = []; Test_1_Valid_Outof_Percentage = [];
Test_2_Valid_Within_Percentage = []; Test_2_Valid_Outof_Percentage = [];
Test_3_Valid_Within_Percentage = []; Test_3_Valid_Outof_Percentage = [];
for i = random_seed
file_name = ['D:\ops\GPclassification\resultsaved\diff_maneuver\track4_maneuver_3_5_10\Random', num2str(i), '_predictedResult.mat'];
load(file_name, 'num_valid_test_1', 'test_1_pred_1_valid_outBoundary', 'test_1_invalid_withinBoundary', 'test_1_invalid_outBoundary', ...
    'num_valid_test_2', 'test_2_pred_1_valid_outBoundary', 'test_2_invalid_withinBoundary', 'test_2_invalid_outBoundary', ...
     'num_valid_test_3', 'test_3_pred_1_valid_outBoundary', 'test_3_invalid_withinBoundary', 'test_3_invalid_outBoundary');
eval(['Random_', num2str(i), '_Test_1_invalid_withinBoundaryPercentage',  '=', ' test_1_invalid_withinBoundary / (test_1_invalid_outBoundary + test_1_invalid_withinBoundary );']);
eval(['Random_', num2str(i), '_Test_1_invalid_OutofBoundaryPercentage',  '=', ' test_1_invalid_outBoundary / (test_1_invalid_outBoundary + test_1_invalid_withinBoundary );']);
eval(['Random_', num2str(i), '_Test_2_invalid_withinBoundaryPercentage',  '=', ' test_2_invalid_withinBoundary / (test_2_invalid_outBoundary + test_2_invalid_withinBoundary );']);
eval(['Random_', num2str(i), '_Test_2_invalid_OutofBoundaryPercentage',  '=', ' test_2_invalid_outBoundary / (test_2_invalid_outBoundary + test_2_invalid_withinBoundary );']);
eval(['Random_', num2str(i), '_Test_3_invalid_withinBoundaryPercentage',  '=', ' test_3_invalid_withinBoundary / (test_3_invalid_outBoundary + test_3_invalid_withinBoundary );']);
eval(['Random_', num2str(i), '_Test_3_invalid_OutofBoundaryPercentage',  '=', ' test_3_invalid_outBoundary / (test_3_invalid_outBoundary + test_3_invalid_withinBoundary );']);


eval(['Random_', num2str(i), '_Test_1_valid_withinBoundaryPercentage',  '=', ' test_1_pred_1_valid_outBoundary / (num_valid_test_1 );']);
eval(['Random_', num2str(i), '_Test_1_valid_OutofBoundaryPercentage',  '=', ' (num_valid_test_1 - test_1_pred_1_valid_outBoundary)  /  (num_valid_test_1 );']);
eval(['Random_', num2str(i), '_Test_2_valid_withinBoundaryPercentage',  '=', ' test_2_pred_1_valid_outBoundary / num_valid_test_2;']);
eval(['Random_', num2str(i), '_Test_2_valid_OutofBoundaryPercentage',  '=', ' (num_valid_test_2 - test_2_pred_1_valid_outBoundary) / num_valid_test_2;']);
eval(['Random_', num2str(i), '_Test_3_valid_withinBoundaryPercentage',  '=', ' test_3_pred_1_valid_outBoundary / num_valid_test_3;']);
eval(['Random_', num2str(i), '_Test_3_valid_OutofBoundaryPercentage',  '=', ' (num_valid_test_3 - test_3_pred_1_valid_outBoundary) / num_valid_test_3;']);




eval(['Test_1_Invalid_Within_Percentage = [Test_1_Invalid_Within_Percentage; Random_', num2str(i), '_Test_1_invalid_withinBoundaryPercentage];' ]);
eval(['Test_1_Invalid_Outof_Percentage = [Test_1_Invalid_Outof_Percentage; Random_', num2str(i), '_Test_1_invalid_OutofBoundaryPercentage];' ]);
eval(['Test_2_Invalid_Within_Percentage = [Test_2_Invalid_Within_Percentage; Random_', num2str(i), '_Test_2_invalid_withinBoundaryPercentage];' ]);
eval(['Test_2_Invalid_Outof_Percentage = [Test_2_Invalid_Outof_Percentage; Random_', num2str(i), '_Test_2_invalid_OutofBoundaryPercentage];' ]);
eval(['Test_3_Invalid_Within_Percentage = [Test_3_Invalid_Within_Percentage; Random_', num2str(i), '_Test_3_invalid_withinBoundaryPercentage];' ]);
eval(['Test_3_Invalid_Outof_Percentage = [Test_3_Invalid_Outof_Percentage; Random_', num2str(i), '_Test_3_invalid_OutofBoundaryPercentage];' ]);


eval(['Test_1_Valid_Within_Percentage = [Test_1_Valid_Within_Percentage; Random_', num2str(i), '_Test_1_valid_withinBoundaryPercentage];' ]);
eval(['Test_1_Valid_Outof_Percentage = [Test_1_Valid_Outof_Percentage; Random_', num2str(i), '_Test_1_valid_OutofBoundaryPercentage];' ]);
eval(['Test_2_Valid_Within_Percentage = [Test_2_Valid_Within_Percentage; Random_', num2str(i), '_Test_2_valid_withinBoundaryPercentage];' ]);
eval(['Test_2_Valid_Outof_Percentage = [Test_2_Valid_Outof_Percentage; Random_', num2str(i), '_Test_2_valid_OutofBoundaryPercentage];' ]);
eval(['Test_3_Valid_Within_Percentage = [Test_3_Valid_Within_Percentage; Random_', num2str(i), '_Test_3_valid_withinBoundaryPercentage];' ]);
eval(['Test_3_Valid_Outof_Percentage = [Test_3_Valid_Outof_Percentage; Random_', num2str(i), '_Test_3_valid_OutofBoundaryPercentage];' ]);

end
Test_1_Invalid_Within_Percentage(isnan(Test_1_Invalid_Within_Percentage)==1) = 0;
Test_1_Invalid_Outof_Percentage(isnan(Test_1_Invalid_Outof_Percentage)==1) = 0;
Test_2_Invalid_Within_Percentage(isnan(Test_2_Invalid_Within_Percentage)==1) = 0;
Test_2_Invalid_Outof_Percentage(isnan(Test_2_Invalid_Outof_Percentage)==1) = 0;
Test_3_Invalid_Within_Percentage(isnan(Test_3_Invalid_Within_Percentage)==1) = 0;
Test_3_Invalid_Outof_Percentage(isnan(Test_3_Invalid_Outof_Percentage)==1) = 0;

Ave_Test_1_Invalid_Within_Percentage = (sum(Test_1_Invalid_Within_Percentage))/num_of_seed;
Ave_Test_1_Invalid_Outof_Percentage = (sum(Test_1_Invalid_Outof_Percentage))/num_of_seed;
Ave_Test_2_Invalid_Within_Percentage = (sum(Test_2_Invalid_Within_Percentage))/num_of_seed;
Ave_Test_2_Invalid_Outof_Percentage = (sum(Test_2_Invalid_Outof_Percentage))/num_of_seed;
Ave_Test_3_Invalid_Within_Percentage = (sum(Test_3_Invalid_Within_Percentage))/num_of_seed;
Ave_Test_3_Invalid_Outof_Percentage = (sum(Test_3_Invalid_Outof_Percentage))/num_of_seed;

Test_1_Valid_Within_Percentage(isnan(Test_1_Valid_Within_Percentage)==1) = 0;
Test_1_Valid_Outof_Percentage(isnan(Test_1_Valid_Outof_Percentage)==1) = 0;
Test_2_Valid_Within_Percentage(isnan(Test_2_Valid_Within_Percentage)==1) = 0;
Test_2_Valid_Outof_Percentage(isnan(Test_2_Valid_Outof_Percentage)==1) = 0;
Test_3_Valid_Within_Percentage(isnan(Test_3_Valid_Within_Percentage)==1) = 0;
Test_3_Valid_Outof_Percentage(isnan(Test_3_Valid_Outof_Percentage)==1) = 0;

Ave_Test_1_Valid_Within_Percentage = (sum(Test_1_Valid_Within_Percentage))/num_of_seed;
Ave_Test_1_Valid_Outof_Percentage = (sum(Test_1_Valid_Outof_Percentage))/num_of_seed;
Ave_Test_2_Valid_Within_Percentage = (sum(Test_2_Valid_Within_Percentage))/num_of_seed;
Ave_Test_2_Valid_Outof_Percentage = (sum(Test_2_Valid_Outof_Percentage))/num_of_seed;
Ave_Test_3_Valid_Within_Percentage = (sum(Test_3_Valid_Within_Percentage))/num_of_seed;
Ave_Test_3_Valid_Outof_Percentage = (sum(Test_3_Valid_Outof_Percentage))/num_of_seed;