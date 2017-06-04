function [acc, loss] = Assignment2()

  [train_X,train_Y,train_y]                   = LoadBatch('data_batch_1.mat');
%   [validation_X,validation_Y,validation_y]    = LoadBatch('data_batch_2.mat');
  [test_X,test_Y,test_y]                      = LoadBatch('test_batch.mat');


  % Different settings for different runs
  % ------------------------------------------------
%   run_variable_index is different for different labs;
  run_variable_index = 5;
  standard_deviation = .001;
  lambda_l        = [ 0,    0,      .1,     1,      0 ];
  n_batch_l       = [ 100,  100,    100,    100,    100 ];
  eta_l           = [ .1,   .01,    .01,    .01,    .1  ];
  n_epochs_l      = [ 40,   40,     40,     40,     200  ];
  rho = {.5,.9,.99};
  decay_rate = .95;

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

  hidden_layer_nodes_1 = 50;

  GDparams = {n_batch, eta, n_epochs};

  disp('Initializing parameters')
  [W_layers, b_layers] = init_param(standard_deviation, hidden_layer_nodes_1, size(train_X,1), size(train_Y,1));

  %  Init momentum to training

  acc = zeros(2,n_epochs);
  loss = zeros(2,n_epochs);
  disp('Starting Mini Batch Gradient Decent')
  fprintf('Epoch = 0');
  
  for i = 1:n_epochs
    [new_W_layers, new_b_layers] = MiniBatchGD(Xtrain, Ytrain, GDparams, W_layers, b_layers, lambda);
    W_layers = new_W_layers;
    b_layers = new_b_layers;

    %   Prints out which epoch you are in
    % ------------------------------------------------
    if i < 11
      fprintf(' \b\b%d', i);
    elseif i < 101
      fprintf(' \b\b\b%d', i);
    else
      fprintf(' \b\b\b\b%d', i);
    end
    % ------------------------------------------------

    loss(1,i) = ComputeCost(Xtrain, Ytrain, W_layers, b_layers, lambda);
    loss(2,i) = ComputeCost(Xtest, test_Y, W_layers, b_layers, lambda);
    acc(1,i) = ComputeAccuracy(Xtrain, train_y, W_layers, b_layers, 'RMSE');
    acc(2,i) = ComputeAccuracy(Xtest, test_y, W_layers, b_layers, 'RMSE');
  end

  fprintf('\n');


end
