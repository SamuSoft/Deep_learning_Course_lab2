function P = EvaluateClassifier(X, W, b)
    P{1} = zeros(50,size(X,2));
    P{2} = zeros(50,size(X,2));
    P{3} = zeros(10,size(X,2));
    for i = 1:size(X,2)
       s1 = W{1}*double(X(:,i)) +b{1};
       P{1}(:,i) = s1;
       h = max(0,s1);
       P{2}(:,i) = h;
       s = W{2}*h + b{2};
       P{3}(:,i) = SOFTMAX(s);
    end

end

function ret = SOFTMAX(P)

    e = exp(P);
    one = ones(size(P,1),1);
    split = one'*e;
    ret = e/split(1,1);
end
