function D = RX(X)

[N M] = size(X);

X_mean = mean(X.').'; %求均值

%X_rep = repmat(X_mean, [1 M]);  %测试C准确性

X = X - repmat(X_mean, [1 M]);

Sigma = (X * X')/M; 

Sigma_inv = inv(Sigma);  %求逆

for m = 1:M
 D(m) = X(:, m)' * Sigma_inv * X(:, m);  % RX表达式
end
