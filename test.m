[train_X,train_Y,train_y]                   = LoadBatch('data_batch_1.mat');
[validation_X,validation_Y,validation_y]    = LoadBatch('data_batch_2.mat');
[test_X,test_Y,test_y]                      = LoadBatch('test_batch.mat');

% Different settings for different runs
% ------------------------------------------------
run_variable_index = 1;
standard_deviation = .001;
lambda_l        = [ 0,    0,      .1,     1,      0 ];
n_batch_l       = [ 100,  100,    100,    100,    100 ];
eta_l           = [ .1,   .01,    .01,    .01,    .1  ];
n_epochs_l      = [ 40,   40,     40,     40,     200  ];
rho = {.5,.9,.99};

lambda = lambda_l(run_variable_index);
n_batch= n_batch_l(run_variable_index);
eta = eta_l(run_variable_index);
n_epochs = n_epochs_l(run_variable_index);
% ------------------------------------------------

Xtrain = double(train_X(:,1:100));
Ytrain = double(train_Y(:,1:100));
Xtest = double(test_X);

%  Preprocessing data
% ------------------------------------------------
mean_X = mean(train_X,2);
Xtrain = double(Xtrain) - repmat(mean_X, [1, size(Xtrain, 2)]);
Xtest = double(test_X) - repmat(mean_X, [1, size(test_X, 2)]);
% ------------------------------------------------

acc = zeros(8,40);
cost = acc;

hidden_layer_nodes_1 = 50;

GDparams = {n_batch, eta, n_epochs};

disp('Initializing parameters')
[W, b] = init_param(standard_deviation, hidden_layer_nodes_1, size(Xtrain,1), size(Ytrain,1));
P = EvaluateClassifier(Xtrain, W, b);
disp('Numeric')
size(Xtrain)
size(P{3})
size(Ytrain)
[grad_W1, grad_b1] = ComputeGradients(Xtrain, Ytrain, P, W, b, lambda)
[grad_b, grad_W] = ComputeGradsNum(Xtrain, Ytrain, W, b, lambda, 10^(-5))
disp(ErrDiff(grad_W{1},grad_W1{1}));
disp(Errdiff(grad_b,grad_b1));
