function acc = ComputeAccuracy(X, y, W, b)
    labeled_data = ones(size(X,2));
    for i = 1:size(X,2)
        p = W*double(X(:,i)) + b;
        [M,I] = max(p);
        labeled_data(i)= I;
    end
    val = 0;
    for i = 1:size(y,1)
        if labeled_data(i) == y(i)
            val = val +1;

        end
    end
    acc = val/size(y,1);

end
