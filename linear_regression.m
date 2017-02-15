%%load file carData.mat
f = matfile('carData.mat');
%% turn matfile to list because matlab is complains about equally space idx
weight = f.weight'; 
mpg = f.mpg';
%% find the length of data set
data_len = length(f.mpg);
%% shuffle data
shuffle_idx = randperm(data_len);

%% get trainset
train_x = weight( shuffle_idx(1:uint64(.7*data_len)));
train_y = mpg( shuffle_idx(1:uint64(.7*data_len)));
%% get test set
test_x = weight( shuffle_idx(uint64(.7*data_len)+1:end));
test_y = mpg( shuffle_idx(uint64(.7*data_len)+1:end));

%% create 1st, 2nd, 3rd order A train matrices
train_a1 = [ ones(length(train_x) ,1), train_x ];
train_a2 = [ ones(length(train_x) ,1), train_x, train_x.^2 ];
train_a3 = [ ones(length(train_x) ,1), train_x, train_x.^2, train_x.^3 ];

%% find a,b,c constants 
%% 1st order
w1 = train_a1\train_y;
%% 2nd order
w2 = train_a2\train_y;
%% 3rd order
w3 = train_a3\train_y;


%% create 1st, 2nd, 3rd order A test matrices
a1_test = [ ones(length(test_x) ,1), test_x];
a2_test = [ ones(length(test_x) ,1), test_x, test_x.^2];
a3_test = [ ones(length(test_x) ,1), test_x, test_x.^2, test_x.^3];

%% 1st order
sse1 = sum((a1_test*w1 - test_y).^2);
%% 2nd order
sse2 = sum((a2_test*w2 - test_y).^2);
%% 3rd order
sse3 = sum((a3_test*w3 - test_y).^2);

%% 1st order
sse4 = sum((train_a1*w1 - train_y).^2);
%% 2nd order
sse5 = sum((train_a2*w2 - train_y).^2);
%% 3rd order
sse6 = sum((train_a3*w3 - train_y).^2);
%% xTest is set is 300 less than min x and 300 greater than max x
%% increments by 5
xTest = [min(weight)-300:5:max(weight)+300]';
%% create test a mat for plotting
a1_plot = [ ones(length(xTest) ,1), xTest];
a2_plot = [ ones(length(xTest) ,1), xTest, xTest.^2];
a3_plot = [ ones(length(xTest) ,1), xTest, xTest.^2, xTest.^3];

disp( strcat('test 1st order sum squared error: ', num2str(sse1)));
disp( strcat('test 2nd order sum squared error: ', num2str(sse2)));
disp( strcat('test 3rd order sum squared error: ', num2str(sse3)));

disp( strcat('train 1st order sum squared error: ', num2str(sse4)));
disp( strcat('train 2nd order sum squared error: ', num2str(sse5)));
disp( strcat('train 3rd order sum squared error: ', num2str(sse6)));

%% plot all three regression lines and the raw data
plot( xTest, a1_plot*w1, xTest, a2_plot*w2, xTest, a3_plot*w3, ...
    train_x, train_y, '.', test_x, test_y, '.');
%% display sse on legend using 4 digits of precision
legend(strcat('test 1st sse:  ',num2str(sse1,4)), ...
strcat('test 2nd sse:  ',num2str(sse2,4)), ...
strcat('test 3rd sse:  ',num2str(sse3,4)),...
'train data','test data');


