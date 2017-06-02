function [W_layers, b_layers] = MiniBatchGD(X, Y, GDparams, W_layers, b_layers, lambda)
    eta = GDparams(2);
    P = EvaluateClassifier(Xtrain, W_layers, b_layers);
    [grad_W_1, grad_b_1] = ComputeGradients(X, Y, P, W_layers, b_layers, lambda);
    W_layers{1} = W_layers{1} - eta*grad_W{1};
    b_layers{1} = b_layers{1} - eta*grad_b{1};
    W_layers{2} = W_layers{2} - eta*grad_W{2};
    b_layers{2} = b_layers{2} - eta*grad_b{2};
end
