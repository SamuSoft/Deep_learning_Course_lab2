function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, b, lambda)
    G = P{3} - Y;

    grad_W{1} = 0;
    grad_W{2} = 0;
    grad_b{1} = 0;
    grad_b{2} = 0;

    for i = 1:size(Y,2)
        grad_b{2} = grad_b{2} + G(:,i);
        grad_W{2} = grad_W{2} + (G(:,i)*P{2}(:,i)');
        g = G(:,i)'*W{2};
        g = g*diag((P{1}(:,i)>0));
        grad_b{1} = grad_b{1} + g;
        grad_W{1} = grad_W{1} + g'*X(:,i)';

    end
    grad_b{1} = grad_b{1}./size(Y,2);
    grad_b{2} = grad_b{2}./size(Y,2);
    grad_W{1} = grad_W{1}./size(Y,2) + 2*lambda*W{1};
    grad_W{2} = grad_W{2}./size(Y,2) + 2*lambda*W{2};
end
