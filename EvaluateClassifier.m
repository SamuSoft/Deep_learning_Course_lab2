function P = EvaluateClassifier(X, W, b)
    P = zeros(10,size(X,2));
    for i = 1:size(X,2)
       s1 = W(1)*double(X(:,i)) +b(1);
       h = max(0,s1);
       s = W(2)*h + b(2);
       P(:,i) = SOFTMAX(s);
    end
end

function ret = SOFTMAX(P)

    e = exp(P);
    one = ones(size(P,1),1);
    split = one'*e;
    ret = e/split(1,1);
end
