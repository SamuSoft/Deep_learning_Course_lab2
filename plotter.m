% b = {ones(size(a{1})),ones(size(a{1})),ones(size(a{1}))};
zero = zeros(1,10);

% for j = 1:700
%     if sum(isnan(a{1}{1,j})) > 0
%         b{1}{j} = a{1}{1,j};
%         b{2}{j} = a{2}{1,j};
%         b{3}{j} = a{3}{1,j};
%     else
%         b{1}{j} = zero;
%         b{2}{j} = zero;
%         b{3}{j} = zero;
%         
%     end
% end
% scatter3(b{1},b{2},b{3});
% xlabel('loss');
% ylabel('lambda');
% zlabel('eta');

hold on
v = zeros(50,1);
for i = 1:50
%         v(i) = (Values{1,1}{i,1}(10) - Values{1}{i,1}(2))/9;
          v(i) = max(Values{1,1}{i,1}(4));
%         scatter3(v(1,:),v(:,j));
%           if v(i) < 3
%               scatter3(Values{1,2}(i),Values{1,3}(i),v(i));
% 
%           end
end
[b,a] = min(v)
val = {Values{1,2}(a),Values{1,3}(a),v(a)};
disp(val{1});
disp(val{2});
disp(val{3});

scatter3(Values{1,2}(a),Values{1,3}(a),v(a));
% scatter3(Values{1,2}(1:50),Values{1,3}(1:50),v);
xlabel('ETA');
ylabel('Lambda');
zlabel('Change Loss');

