function J = ComputeCost(X, Y, W, b, lambda)
    % Construction of D
    % D(i) = [X(:,i),y(i)]
    % size(D) = [3072x1]x100000
    
    P = EvaluateClassifier(X,W,b);

    sum_p = 0;
    for i = 1:size(Y,2)
        sum_p = sum_p + l_cross(Y(:,i),P(:,i));
    end
    J = 1/size(X,2)*sum_p + lambda*sum(sum(W.^2));
    
    
% each column of X corresponds to an image and X has size d?n.
% each column of Y (K?n) is the one-hot ground truth label for the corre- 
% sponding column of X or Y is the (1?n) vector of ground truth labels.
% J is a scalar corresponding to the sum of the loss of the network?s 
% predictions for the images in X relative to the ground truth labels and the 
% regularization term on W.
end

function ret = l_cross(y,p)
    ret = -log(y'*p);
end
