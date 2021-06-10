%% AutoEncoder Collect Data
% Two Tracks independent
% Two Inputs and their corresponding outputs
% The code is similar to Test-07
close all; clear; 
rng(3) % used 3 before 10.3.2020
%% step1: collect different magnitude maneuvers
km = 1000;
length = 4; % track length
gap = 0; % track gap for every two points
measure_gap = 1; % 
database_name = {'results-20201111-0.0050N-300s1500Isp_5sigma';
                               'results-20201111-0.0100N-300s1500Isp_5sigma';
                               'results-20201111-0.0200N-50s1500Isp_5sigma';
                               'results-20201111-0.0200N-100s1500Isp_5sigma';
                               'results-20201111-0.0200N-500s1500Isp_5sigma';
                               'results-20201111-0.2000N-300s1500Isp_5sigma';
                               'results-20201111-0.5000N-300s1500Isp_5sigma';
                               'results-20201111-1.0000N-300s1500Isp_5sigma';
                               'results-20201111-10.0000N-300s1500Isp_5sigma';
                               'results-20201111-50.0000N-300s1500Isp_5sigma'; %10
                               'results-20201019-1mps_5sigma';
                               'results-20201019-2mps_5sigma';
                               'results-20201019-3mps_5sigma';
                               'results-20201019-4mps_5sigma';
                               'results-20201019-5mps_5sigma';
                               'results-20201019-6mps_5sigma';
                               'results-20201019-7mps_5sigma';
                               'results-20201019-8mps_5sigma';
                               'results-20201019-9mps_5sigma';
                               'results-20201019-10mps_5sigma';};
maneuver = [4 6 7 13 16 18];
unknown_maneuver = [1 3 5 8 11 15 19 20];
 part_size = 150;
% maneuver = [14];
% unknown_maneuver = [15];
%  part_size = 1;
% maneuver = [4 6 8];
% unknown_maneuver = [1 3 5 9];
%  part_size = 120;
%  
 training_input_1_data = []; training_input_2_data = []; training_output_data = [];
testing_1_input_1_data = []; testing_1_input_2_data = []; testing_1_output_data = [];
testing_2_input_1_data = []; testing_2_input_2_data = []; testing_2_output_data = [];
testing_3_input_1_data = []; testing_3_input_2_data = []; testing_3_output_data = [];

%% Normalize the training data
total_train_x = []; total_train_y = []; total_train_z = [];
for maneu = maneuver
    file_name = database_name(maneu);
    file_name = char(file_name);
    file_path = ['D:\ops\GPclassification\maneuver_database\RobustnessDataBase\', file_name, '.mat'];
    load(file_path);
    time_point = measures(:,2);
%     database = measures(:, [18:20]); % with noise
   database = measures(:, [6:8]); % without noise
     training_range = 5;
    t1 = 1; 
    t2 = 28 * training_range;
    n1 = find(measures(:,2)<(t1*86400));
    n2 = find(measures(:,2)<(t2*86400));
    n1 = size(n1); n1 = n1(1);
    n2 = size(n2); n2 = n2(1);
    training_data = database(n1:n2, :);
    train_x = training_data(:,1);
    train_y = training_data(:,2);
    train_z = training_data(:,3);
    eval(['total_train_x' '=' '[total_train_x;', 'train_x];']);
    eval(['total_train_y' '=' '[total_train_y;', 'train_y];']);
    eval(['total_train_z' '=' '[total_train_z;', 'train_z];']);    
end
    % basic parameter
    a = 0; % lower
    b = 1; % higher
    Xmax = max(total_train_x);
    Xmin = min(total_train_x);
    kx = (b - a)./(Xmax - Xmin);
    
    Ymax = max(total_train_y);
    Ymin = min(total_train_y);
    ky = (b - a)./(Ymax - Ymin);
    
    Zmax=max(total_train_z);
    Zmin=min(total_train_z);
    kz=(b-a)./(Zmax-Zmin);
    

%% Combine the data structure into Tensor
for maneu = maneuver
     file_path = ['D:\ops\GPclassification\maneuver_database\RobustnessDataBase\', file_name, '.mat'];
    load(file_path);
    time_point = measures(:,2);
    %     database = measures(:, [18:20]); % with noise
   database = measures(:, [6:8]); % without noise
    all_test_tensor_1=[]; all_test_tensor_2=[];
    
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
    
    train_nor_x = a + kx * (train_x - Xmin);
    test_nor_x = a + kx * (test_x - Xmin);

    train_nor_y = a + ky * (train_y - Ymin);
    test_nor_y = a + ky * (test_y - Ymin);

    train_nor_z = a+kz * (train_z - Zmin);
    test_nor_z = a + kz * (test_z - Zmin);

    train_nor = [train_nor_x  train_nor_y  train_nor_z];
    test_nor = [test_nor_x test_nor_y test_nor_z];
    data_nor = [train_nor; test_nor];
    
        % collect data
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
                    part_track_tensor_1(i, :, :) = part_data_1;
%                     part_track_tensor_1 = part_track_tensor_1(:)';
                    part_track_tensor_2(i, :, :) = part_data_2;
%                     part_track_tensor_2 = part_track_tensor_2(:)';
                    eval(['first_orbit_', num2str(orbit), '=[','first_orbit_', num2str(orbit), ';part_track_tensor_1(i, :, :)];']);
                    eval(['second_orbit_', num2str(orbit), '=[','second_orbit_', num2str(orbit), ';part_track_tensor_2(i, :, :)];']);
                end
            else
            end
        end   
    end
    
    
    % Combine the data
    train_nor_data_1 = []; train_nor_data_2=[]
    for i = 1:training_range
        eval(['orbit_size' '=' 'size(first_orbit_', num2str(i), ');']);
       orbit_size = orbit_size(1); 
%        each_train_size = round(orbit_size/part_size);
%        select_index = randperm(orbit_size, each_train_size);
       eval(['select_1_train_', num2str(i), '=' 'first_orbit_', num2str(i),'(1:part_size:end, :, :);']);
       eval(['select_2_train_', num2str(i), '=' 'second_orbit_', num2str(i),'(1:part_size:end, :, :);']);       
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
    train_output_0 = zeros(train_size_0, 1);

    input_1 = [first_input_1; first_input_0];
    input_2 = [second_input_1; second_input_0];
    output_data = [train_output_1; train_output_0];

    training_size = size(output_data); training_size = training_size(1);
    random_index = randperm(training_size, training_size);
    input_1 = input_1(random_index, :, :);
    input_2 = input_2(random_index, :, :);
    train_output = output_data(random_index, :);
%     train_input_ = [input_1 input_2];
    

    eval(['train_1_in_maneu_', num2str(maneu), '= input_1;']);
    eval(['train_2_in_maneu_', num2str(maneu), '= input_2;']);
    eval(['train_out_maneu_', num2str(maneu), '=', 'train_output;']);



    % test data 1
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
    test_output_0 = zeros(test_size_0, 1);

    test_input_1 = [first_test_1; first_test_0];
    test_input_2 = [second_test_1; second_test_0];
    test_output_data = [test_output_1; test_output_0];

    test_size = size(test_output_data); test_size = test_size(1);
    random_index = randperm(test_size, test_size);
    test_input_1 = test_input_1(random_index, :, :);
    test_input_2 = test_input_2(random_index, :, :);

    test_data_out_1 = test_output_data(random_index, :);
    test_1_input_1 = test_input_1; 
    test_1_input_2 = test_input_2;
    
    %test data 2
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
    test_output_0 = zeros(test_size_0, 1);

    test_input_1 = [first_test_1; first_test_0];
    test_input_2 = [second_test_1; second_test_0];
    test_output_data = [test_output_1; test_output_0];

    test_size = size(test_output_data); test_size = test_size(1);
    random_index = randperm(test_size, test_size);
    test_input_1 = test_input_1(random_index, :, :);
    test_input_2 = test_input_2(random_index, :, :);

    test_data_out_2 = test_output_data(random_index, :);
    test_2_input_1 = test_input_1; 
    test_2_input_2 = test_input_2;
    
    eval(['test_1_in_1_maneu_', num2str(maneu), '=', 'test_1_input_1;']);
    eval(['test_1_in_2_maneu_', num2str(maneu), '=', 'test_1_input_2;']);
    eval(['test_1_out_maneu_', num2str(maneu), '=', 'test_data_out_1;']);
    eval(['test_2_in_1_maneu_', num2str(maneu), '=', 'test_2_input_1;']);
    eval(['test_2_in_2_maneu_', num2str(maneu), '=', 'test_2_input_2;']);
    eval(['test_2_out_maneu_', num2str(maneu), '=', 'test_data_out_2;']);
    
    eval(['training_input_1_data', '=', '[training_input_1_data;', 'train_1_in_maneu_', num2str(maneu), '];']);
     eval(['training_input_2_data', '=', '[training_input_2_data;', 'train_2_in_maneu_', num2str(maneu), '];']);
    eval(['training_output_data', '=', '[training_output_data;', 'train_out_maneu_', num2str(maneu), '];']);
    eval(['testing_1_input_1_data', '=', '[testing_1_input_1_data;', 'test_1_in_1_maneu_', num2str(maneu), '];']);
    eval(['testing_1_input_2_data', '=', '[testing_1_input_2_data;', 'test_1_in_2_maneu_', num2str(maneu), '];']);
    eval(['testing_1_output_data', '=', '[testing_1_output_data;', 'test_1_out_maneu_', num2str(maneu), '];']);
    eval(['testing_2_input_1_data', '=', '[testing_2_input_1_data;', 'test_2_in_1_maneu_', num2str(maneu), '];']);
     eval(['testing_2_input_2_data', '=', '[testing_2_input_2_data;', 'test_2_in_2_maneu_', num2str(maneu), '];']);
    eval(['testing_2_output_data', '=', '[testing_2_output_data;', 'test_2_out_maneu_', num2str(maneu), '];']);
end

%% Collect the data with unknown maneuver with noise
for maneu = unknown_maneuver
    file_name = database_name(maneu);
    file_name = char(file_name);
    file_path = ['D:\ops\GPclassification\maneuver_database\RobustnessDataBase\', file_name, '.mat'];
    load(file_path);
    time_point = measures(:,2);
    database = measures(:, [18:20]);
    all_test_tensor_1 =[]; all_test_tensor_2=[];
    % normalize the data by using training parameter
    test_data = database;
    test_x = test_data(:,1);
    test_y = test_data(:,2);
    test_z = test_data(:,3);
    test_nor_x = a + kx * (test_x - Xmin);
    test_nor_y = a + ky * (test_y - Ymin);
    test_nor_z = a + kz * (test_z - Zmin);
    test_nor = [test_nor_x test_nor_y test_nor_z];
    data_nor = test_nor;
    % collect the orbit data
    for orbit = 1:10
        t1 = 1 + 28 * (orbit -1);
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
                    part_track_tensor_1(i, :, :) = part_data_1;
                    part_track_tensor_2(i, :, :) = part_data_2;
                    eval(['first_orbit_', num2str(orbit), '=[','first_orbit_', num2str(orbit), ';part_track_tensor_1(i, :, :)];']);
                    eval(['second_orbit_', num2str(orbit), '=[','second_orbit_', num2str(orbit), ';part_track_tensor_2(i, :, :)];']); end
            else
            end
        end
        
    end
    
    % combine the data
    % test data 3
    test_nor_data_1 = []; test_nor_data_2 = [];
    for i = 1:10
        eval(['test_nor_data_1' '=' '[test_nor_data_1;', 'first_orbit_', num2str(i), '];']);
        eval(['test_nor_data_2' '=' '[test_nor_data_2;', 'second_orbit_', num2str(i), '];'])    
    end
    first_test_1 = test_nor_data_1;
    second_test_1 = test_nor_data_2;
    test_size_1 = size(first_test_1); test_size_1 = test_size_1(1);
    test_output_1 = ones(test_size_1, 1);

    test_nor_data_1 = []; test_nor_data_2 = [];
    normal_index = [1,2,3,4,5,6,7,8,9,10];
    fix_index = [2,3,4,5,6,7,8,9,10,1];
    for i = 1:10
        normal_order = normal_index(i);
        fix_order = fix_index(i);
        eval(['test_nor_data_1' '=' '[test_nor_data_1;', 'first_orbit_', num2str(normal_order), '];']);
        eval(['test_nor_data_2' '=' '[test_nor_data_2;', 'first_orbit_', num2str(fix_order), '];']);
    end
    first_test_0 = test_nor_data_1;
    second_test_0 = test_nor_data_2;
    test_size_0 = size(first_test_0); test_size_0 = test_size_0(1);
    test_output_0 = zeros(test_size_0, 1);

    test_input_1 = [first_test_1; first_test_0];
    test_input_2 = [second_test_1; second_test_0];
    test_output_data = [test_output_1; test_output_0];

    test_size = size(test_output_data); test_size = test_size(1);
    random_index = randperm(test_size, test_size);
    test_input_1 = test_input_1(random_index, :, :);
    test_input_2 = test_input_2(random_index, :, :);

    test_data_out_3 = test_output_data(random_index, :);
    test_3_data_input_1 = test_input_1;
    test_3_data_input_2 = test_input_2;
    
    eval(['test_3_in_1_maneu_', num2str(maneu), '=', 'test_3_data_input_1;']);
    eval(['test_3_in_2_maneu_', num2str(maneu), '=', 'test_3_data_input_2;']);
    eval(['test_3_out_maneu_', num2str(maneu), '=', 'test_data_out_3;']);
    eval(['testing_3_input_1_data', '=', '[testing_3_input_1_data;', 'test_3_in_1_maneu_', num2str(maneu), '];']);
    eval(['testing_3_input_2_data', '=', '[testing_3_input_2_data;', 'test_3_in_2_maneu_', num2str(maneu), '];']);
    eval(['testing_3_output_data', '=', '[testing_3_output_data;', 'test_3_out_maneu_', num2str(maneu), '];']);
    
end
%% Save the data
save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\training_input_1_data.mat', 'training_input_1_data');
save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\training_input_2_data.mat', 'training_input_2_data');
save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\training_output_data.mat', 'training_output_data');

save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_1_input_1_data.mat', 'testing_1_input_1_data');
save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_1_input_2_data.mat', 'testing_1_input_2_data');
save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_1_output_data.mat', 'testing_1_output_data');

save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_2_input_1_data.mat', 'testing_2_input_1_data');
save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_2_input_2_data.mat', 'testing_2_input_2_data');
save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_2_output_data.mat', 'testing_2_output_data');

save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_3_input_1_data.mat', 'testing_3_input_1_data');
save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_3_input_2_data.mat', 'testing_3_input_2_data');
save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\testing_3_output_data.mat', 'testing_3_output_data');

save('D:\ops\GPclassification\CollectDatabase\AutoEncoder\TwoIndependent\AllData.mat');

