function Assignment2()

    [train_X,train_Y,train_y]                   = LoadBatch('data_batch_1.mat');
    [validation_X,validation_Y,validation_y]    = LoadBatch('data_batch_2.mat');
    [test_X,test_Y,test_y]                      = LoadBatch('test_batch.mat');


    % Different settings for different runs
    % ------------------------------------------------
    run_variable_index = 1;
    standard_deviation = .001;
    lambda_l        = [ 0, 0, .1, 1 ];
    n_batch_l       = [ 100,  100,  100,  100 ];
    eta_l           = [ .1, .01,  .01,  .01  ];
    n_epochs_l      = [ 40, 40, 40, 40  ];

    lambda = lambda_l(run_variable_index);
    n_batch= n_batch_l(run_variable_index);
    eta = eta_l(run_variable_index);
    n_epochs = n_epochs_l(run_variable_index);
    % ------------------------------------------------

     Xtrain = train_X;
     Ytrain = train_Y;

    %  Preprocessing data
    % ------------------------------------------------
     mean_X = mean(Xtrain,2);
     Xtrain = double(Xtrain) - repmat(mean_X, [1, size(Xtrain, 2)]);
    % ------------------------------------------------

     acc = zeros(8,40);
     cost = acc;

     hidden_layer_nodes_1 = 50;

     GDparams = [n_batch, eta, n_epochs];

     [W_layers, b_layers] = init_param(standard_deviation, hidden_layer_nodes_1, size(train_X,1), size(train_Y,1));

     P = EvaluateClassifier(Xtrain(:,1), W_layers, b_layers);
    %  for i = 1:size(Y,2)
     %
    %  end


     [grad_W, grad_b] = ComputeGradients(Xtrain(:,1), Ytrain(:,1), P, W_layers, b_layers, lambda);
    %  [grad_W_1, grad_b_1] = ComputeGradients(Xtrain, Ytrain, P{1}, W_layers{1}, b_layers{1}, lambda);

    %  Packaging the layers
    %  W_layers = {W_1,W_2};
    %  b_layers = {b_1,b_2};

    %  EvaluateClassifier(Xtrain(:,1), W_layers, b_layers);
    %  ComputeCost(Xtrain, Ytrain, W_layers, b_layers, lambda);


     %
    %  for i = 1:n_epochs
     %
    %      for j=1:size(Ytrain,2)/n_batch
    %         j_start = (j-1)*n_batch + 1;
    %         j_end = j*n_batch;
    %         inds = j_start:j_end;
    %         Xbatch = Xtrain(:, j_start:j_end);
    %         Ybatch = Ytrain(:, j_start:j_end);
    %         [W,b] = MiniBatchGD(Xbatch, Ybatch, GDparams, W, b, lambda);
    %      end
    %      acc(k,i) = ComputeAccuracy(test_X, test_y, W, b);
    %      acc(k+4,i) = ComputeAccuracy(train_X, train_y, W, b);
     %
    %      cost(k,i) =  ComputeCost(test_X, test_Y, W, b, lambda);
    %      cost(k+4,i) = ComputeCost(train_X, train_Y, W, b,lambda);
    %  end
end
