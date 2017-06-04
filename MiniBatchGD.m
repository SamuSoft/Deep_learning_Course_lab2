function [W_layers, b_layers] = MiniBatchGD(X, Y, GDparams, W_layers, b_layers, lambda)
    N = size(X,2);
    n_batch = GDparams{1};
    eta = GDparams{2};
    n_epochs = GDparams{3};
%     GDparams = {n_batch, eta, n_epochs}

%     for i = 1:n_epochs
    for j = 1:N/n_batch
%             Sets the parts of the dataset for this batch
        Xtrain = X(:,((j-1)*n_batch + 1):(j*n_batch));
        Ytrain = Y(:,((j-1)*n_batch + 1):(j*n_batch));
        P = EvaluateClassifier(Xtrain, W_layers, b_layers);
        [grad_W, grad_b] = ComputeGradients(Xtrain, Ytrain, P, W_layers, b_layers, lambda);
    %     Update Layer 1
        W_layers{1} = W_layers{1} - eta*grad_W{1};
        b_layers{1} = b_layers{1} - eta*grad_b{1};
    %     Update Layer 2
        W_layers{2} = W_layers{2} - eta*grad_W{2};
        b_layers{2} = b_layers{2} - eta*grad_b{2};
    end
%     end
end
