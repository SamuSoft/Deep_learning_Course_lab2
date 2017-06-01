function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, b, lambda)
    G = P - Y;
    grad_W = 0;
    grad_b = 0;
    for i = 1:size(Y,2)
        grad_b = grad_b + G(:,i);
        grad_W = grad_W + G(:,i)*double(X(:,i))';
    end
    grad_b = grad_b./size(Y,2);
    grad_W = grad_W./size(Y,2) + 2*lambda*W;
end