function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
    eta = GDparams(2);
    P = EvaluateClassifier(X,W,b);
    [grad_W, grad_b] = ComputeGradients(X, Y,P , W, b, lambda);
    Wstar = W - eta*grad_W;
    bstar = b - eta*grad_b;
end