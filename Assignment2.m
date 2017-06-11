

  [train_X,train_Y,train_y]                   = LoadBatch('data_batch_1.mat');
%   [validation_X,validation_Y,validation_y]    = LoadBatch('data_batch_2.mat');
  [test_X,test_Y,test_y]                      = LoadBatch('test_batch.mat');


  % Different settings for different runs
  % ------------------------------------------------
%   run_variable_index is different for different labs;
  run_variable_index = 5;
  standard_deviation = .001;
  lambda_l        = [ .000001,    .000001,      .000001,     .000001,      .000001 ];
  n_batch_l       = [ 100,  100,    100,    100,    100 ];
  eta_l           = [ .1,   .01,    .01,    .01,    .01  ];
  n_epochs_l      = [ 5,   5,     5,     5,     40  ];
  rho_l = {.5,.9,.99,.5,.9,.99};
  rho = .9;%rho(run_variable_index);
  decay_rate = .95;
  % lambda_l        = [ ,    ,      ,     ,       ];
  % n_batch_l       = [ ,    ,      ,     ,       ];
  % eta_l           = [ ,    ,      ,     ,       ];
  % n_epochs_l      = [ ,    ,      ,     ,       ];

  lambda = lambda_l(run_variable_index);
  n_batch= n_batch_l(run_variable_index);
  eta = eta_l(run_variable_index);
  n_epochs = n_epochs_l(run_variable_index);
  hidden_layer_nodes_1 = 50;
  GDparams = {n_batch, eta, n_epochs, rho, standard_deviation, lambda, hidden_layer_nodes_1};
  % ------------------------------------------------

  Xtrain = double(train_X);
  Ytrain = double(train_Y);
  ytrain = double(train_y);
  Xtest = double(test_X);

  %  Preprocessing data
  % ------------------------------------------------
  mean_X = mean(train_X,2);
  Xtrain = double(Xtrain) - repmat(mean_X, [1, size(Xtrain, 2)]);
  Xtest = double(test_X) - repmat(mean_X, [1, size(test_X, 2)]);
  % ------------------------------------------------

  Train_Data = {Xtrain(:,1:9000), Ytrain(:,1:9000), ytrain(1:9000,1)};
  Test_Data = {Xtest(:,1:1000),test_Y(:,1:1000),test_y(1:1000,1)};

  loops = 1;
  e_min = -10;
  e_max = log(2);
  l_min = -10;
  l_max = log(6);
  % loss_list = zeros(1,loops);
  
  loss_list = cell(50,50);
  top_value = 10;
  loops = 200;
  top_val = top_value/loops;
  eta_list = 0:top_val:10;%zeros(1,loops);
  lambda_list = 0:top_val:10;%zeros(1,loops);

%   for j = 1:loops
%     for i = 1:loops
%       eta = eta_list(i);
%       lambda = lambda_list(j);
%       GDparams = {n_batch, eta, 5, .9, standard_deviation, lambda, 50};
%       [Data, ~] = RunNetwork(Train_Data, Test_Data, GDparams);
%        loss_list{j,i} = squeeze(Data{2});
%      end
%    end
%    Values = {loss_list, lambda_list, eta_list};

  for i = 1:1
      % random eta
      e = e_min + (e_max-e_min)*rand();
      eta = 10^e;
      l = l_min + (l_max-l_min)*rand();
      lambda = 10^l;
      lambda = 3.2764e-06;

       eta= 4.1980e-04;
  
%       eta = .01;
%       lambda = .000001;
%       rho = .9;
      n_epochs = 30;
%       hidden_layer_nodes_1 = 50;
      GDparams = {n_batch, eta, n_epochs, rho, standard_deviation, lambda, hidden_layer_nodes_1};
      [Data, Model] = RunNetwork(Train_Data, Test_Data, GDparams);
      loss_list_test = Data{2}(2,:);
      loss_list_train = Data{2}(1,:);
      lambda_list(1,i) = lambda;
      eta_list(1,i) = eta;
      disp('Loop: ');
      disp(i);
  end
  Values = {loss_list_test, loss_list_train};

