%% Two track model classification with two independent inputs
% Training  data is from measurement data
% Test data is from measurement data
clear size;
close all;
clear
load('D:\maneuver_detection\simulation_java\co_paper04_spacops-master\co_paper04_spacops-master\Java\results-20201019-1N-50s.mat')
km = 1000;
length = 15; % track length
gap = 0; % track gap for every two points
measure_gap = 1; % 
time_point = measures(:,2);
database = measures(:, [18:20]);
all_test_tensor_1=[]; all_test_tensor_2=[];
rng(3)

% normalize the database
training_range = 5;
t1 = 1; 
t2 = 28 * training_range;
t3 = 28* (training_range +1);
t4 = 28 * 10;

n1 = find(measures(:,2)<(t1*86400));
n2 = find(measures(:,2)<(t2*86400));
n1 = size(n1); n1 = n1(1);
n2 = size(n2); n2 = n2(1);

n3 = find(measures(:,2)<(t3*86400));
n4 = find(measures(:,2)<(t4*86400));
n3 = size(n3); n3 = n3(1);
n4 = size(n4); n4 = n4(1);
training_data = database(1:n2, :);
train_x = training_data(:,1);
train_y = training_data(:,2);
train_z = training_data(:,3);
test_data = database(n2+1:end, :);
test_x = test_data(:,1);
test_y = test_data(:,2);
test_z = test_data(:,3);
% basic parameter
a=0; % lower
b=1; % higher

Xmax = max(train_x);
Xmin = min(train_x);
kx = (b - a)./(Xmax - Xmin);
train_nor_x = a + kx * (train_x - Xmin);
test_nor_x = a + kx * (test_x - Xmin);

Ymax = max(train_y);
Ymin = min(train_y);
ky = (b - a)./(Ymax - Ymin);
train_nor_y = a + ky * (train_y - Ymin);
test_nor_y = a + ky * (test_y - Ymin);

Zmax=max(train_z);
Zmin=min(train_z);
kz=(b-a)./(Zmax-Zmin);
train_nor_z = a+kz * (train_z - Zmin);
test_nor_z = a + kz * (test_z - Zmin);

train_nor = [train_nor_x  train_nor_y  train_nor_z];
test_nor = [test_nor_x test_nor_y test_nor_z];
data_nor = [train_nor; test_nor];

%% collect data
for orbit = 1:10
    t1 =1+28*(orbit-1);
    t2 = 28*orbit;
    n1 = find(measures(:,2)<(t1*86400));
    n2 = find(measures(:,2)<(t2*86400));
    n1 = size(n1); 
    n2 = size(n2); 
    n2 = n2(1);n1 = n1(1);
    tt = time_point(n1:n2,:);
    y=diff(tt);
    k = find(y>15);
    n2 = n2(1);n1 = n1(1);
    n = n2 - n1;
    m = size(k);
    m = m(1);
    mea_orbit_size = fix(n/(length*(gap+1))) - 1; % get the number of track with length M in orbit
    mea_orbit_data = data_nor([n1:n2], :)/km;
    orbit_name = ['first_orbit_', num2str(orbit)];
    eval(['first_orbit_', num2str(orbit), '=', '[];',]);
    orbit_name = ['second_orbit_', num2str(orbit)];
    eval(['second_orbit_', num2str(orbit), '=', '[];',]);
    for i = 2:m
        data_size(i)=k(i)-k(i-1);
        if data_size(i) > length+1
            variable = eval(['data_nor((k(i-1)+n1-1):(k(i)+n1-1),:)', ';']); 
            for n = 1:data_size(i)-length-1
                part_data_1 = variable(n:n+length-1, :);
                part_data_2 = variable(n+1:n+length, :);
                part_track_tensor_1 = part_data_1';
                part_track_tensor_1 = part_track_tensor_1(:)';
                part_track_tensor_2 = part_data_2';
                part_track_tensor_2 = part_track_tensor_2(:)';
                eval(['first_orbit_', num2str(orbit), '=[','first_orbit_', num2str(orbit), ';part_track_tensor_1];']);
                eval(['second_orbit_', num2str(orbit), '=[','second_orbit_', num2str(orbit), ';part_track_tensor_2];']);
            end

        else
        end
    end   
end
%% Combine the data
train_nor_data_1 = []; train_nor_data_2 = [];
part_size = 50;
for i = 1:training_range
    eval(['orbit_size' '=' 'size(first_orbit_', num2str(i), ');']);
   orbit_size = orbit_size(1); 
   each_train_size = round(orbit_size/part_size);
   select_index = randperm(orbit_size, each_train_size);
   eval(['select_1_train_', num2str(i), '=' 'first_orbit_', num2str(i),'(select_index, :);']);
   eval(['select_2_train_', num2str(i), '=' 'second_orbit_', num2str(i),'(select_index, :);']);
   eval(['train_nor_data_1' '=' '[train_nor_data_1;', 'select_1_train_', num2str(i), '];']);
   eval(['train_nor_data_2' '=' '[train_nor_data_2;', 'select_2_train_', num2str(i), '];'])    
end
first_input_1 = train_nor_data_1;
second_input_1 = train_nor_data_2;
train_size_1 = size(first_input_1); train_size_1 = train_size_1(1);
train_output_1 = ones(train_size_1, 1);

train_nor_data_1 = []; train_nor_data_2 = [];
normal_index = [1,2,3,4,5];
fix_index = [2,3,4,5,1];
for i = 1:training_range
    normal_order = normal_index(i);
    fix_order = fix_index(i);
    eval(['train_nor_data_1' '=' '[train_nor_data_1;', 'select_1_train_', num2str(normal_order), '];']);
    eval(['train_nor_data_2' '=' '[train_nor_data_2;', 'select_2_train_', num2str(fix_order), '];']);
end
first_input_0 = train_nor_data_1;
second_input_0 = train_nor_data_2;
train_size_0 = size(first_input_0); train_size_0 = train_size_0(1);
train_output_0 = -ones(train_size_0, 1);

input_1 = [first_input_1; first_input_0];
input_2 = [second_input_1; second_input_0];
output_data = [train_output_1; train_output_0];

training_size = size(output_data); training_size = training_size(1);
random_index = randperm(training_size, training_size);
input_1 = input_1(random_index, :, :);
input_2 = input_2(random_index, :, :);
train_output = output_data(random_index, :);
train_input = [input_1 input_2];


%% test data 1
test_nor_data_1 = []; test_nor_data_2 = [];
for i = 1:training_range
    eval(['test_nor_data_1' '=' '[test_nor_data_1;', 'first_orbit_', num2str(i), '];']);
    eval(['test_nor_data_2' '=' '[test_nor_data_2;', 'second_orbit_', num2str(i), '];'])    
end
first_test_1 = test_nor_data_1;
second_test_1 = test_nor_data_2;
test_size_1 = size(first_test_1); test_size_1 = test_size_1(1);
test_output_1 = ones(test_size_1, 1);

test_nor_data_1 = []; test_nor_data_2 = [];
normal_index = [1,2,3,4,5];
fix_index = [2,3,4,5,1];
for i = 1:training_range
    normal_order = normal_index(i);
    fix_order = fix_index(i);
    eval(['test_nor_data_1' '=' '[test_nor_data_1;', 'first_orbit_', num2str(normal_order), '];']);
    eval(['test_nor_data_2' '=' '[test_nor_data_2;', 'first_orbit_', num2str(fix_order), '];']);
end
first_test_0 = test_nor_data_1;
second_test_0 = test_nor_data_2;
test_size_0 = size(first_test_0); test_size_0 = test_size_0(1);
test_output_0 = -ones(test_size_0, 1);

test_input_1 = [first_test_1; first_test_0];
test_input_2 = [second_test_1; second_test_0];
test_output_data = [test_output_1; test_output_0];

test_size = size(test_output_data); test_size = test_size(1);
random_index = randperm(test_size, test_size);
test_input_1 = test_input_1(random_index, :, :);
test_input_2 = test_input_2(random_index, :, :);

test_data_out_1 = test_output_data(random_index, :);
test_data_in_1 = [test_input_1 test_input_2];
%% test data 2
test_nor_data_1 = []; test_nor_data_2 = [];
for i = training_range+1:10
    eval(['test_nor_data_1' '=' '[test_nor_data_1;', 'first_orbit_', num2str(i), '];']);
    eval(['test_nor_data_2' '=' '[test_nor_data_2;', 'second_orbit_', num2str(i), '];'])    
end
first_test_1 = test_nor_data_1;
second_test_1 = test_nor_data_2;
test_size_1 = size(first_test_1); test_size_1 = test_size_1(1);
test_output_1 = ones(test_size_1, 1);

test_nor_data_1 = []; test_nor_data_2 = [];
normal_index = [6,7,8,9,10];
fix_index = [7,8,9,10,6];
for i = training_range+1:10
    normal_order = normal_index(i-5);
    fix_order = fix_index(i-5);
    eval(['test_nor_data_1' '=' '[test_nor_data_1;', 'first_orbit_', num2str(normal_order), '];']);
    eval(['test_nor_data_2' '=' '[test_nor_data_2;', 'first_orbit_', num2str(fix_order), '];']);
end
first_test_0 = test_nor_data_1;
second_test_0 = test_nor_data_2;
test_size_0 = size(first_test_0); test_size_0 = test_size_0(1);
test_output_0 = -ones(test_size_0, 1);

test_input_1 = [first_test_1; first_test_0];
test_input_2 = [second_test_1; second_test_0];
test_output_data = [test_output_1; test_output_0];

test_size = size(test_output_data); test_size = test_size(1);
random_index = randperm(test_size, test_size);
test_input_1 = test_input_1(random_index, :, :);
test_input_2 = test_input_2(random_index, :, :);

test_data_out_2 = test_output_data(random_index, :);
test_data_in_2 = [test_input_1 test_input_2];

%% Save the data
save('D:\ops\GPclassification\CollectDatabase\ThrustData\1N-50s\train_input.mat', 'train_input');
save('D:\ops\GPclassification\CollectDatabase\ThrustData\1N-50s\train_output.mat', 'train_output');
save('D:\ops\GPclassification\CollectDatabase\ThrustData\1N-50s\test_data_in_1.mat', 'test_data_in_1');
save('D:\ops\GPclassification\CollectDatabase\ThrustData\1N-50s\test_data_out_1.mat', 'test_data_out_1');
save('D:\ops\GPclassification\CollectDatabase\ThrustData\1N-50s\test_data_in_2.mat', 'test_data_in_2');
save('D:\ops\GPclassification\CollectDatabase\ThrustData\1N-50s\test_data_out_2.mat', 'test_data_out_2');

save('D:\ops\GPclassification\CollectDatabase\ThrustData\1N-50s\nor_data_parameter.mat', 'Xmin', 'Xmax', 'Ymin', 'Ymax', 'Zmin', 'Zmax', 'kx', 'ky', 'kz', 'a', 'b');
save('D:\ops\GPclassification\CollectDatabase\ThrustData\1N-50s\AllData.mat');