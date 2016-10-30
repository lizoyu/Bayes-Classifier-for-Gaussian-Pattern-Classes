function [ d21 ] = GaussianBayes( n, N, m1, m2 )
% Bayes classifier for Gaussian pattern classes
% - use training data to form the decision function
% input:
% 	n: the amount of features (dimensions)
%	N: the amount of pattern vectors (assumed both classes have the same amout of training data)
%	m1: mean vector of class 1 (mean of the training data)
%	m2: mean vector of class 2 (mean of the training data)
C1 = zeros(n);
C2 = C1(:);
for i = 1:N
    C1 = C1 + e1(:,i) * e1(:,i)';
    C2 = C2 + e2(:,i) * e2(:,i)';
end
C1 = C1/N - m1 * m1';
C2 = C2/N - m2 * m2';

syms x1 x2 x3 x4 x5 x6
x = [x1;x2;x3;x4;x5;x6];
xt = [x1,x2,x3,x4,x5,x6];
d1 = log(0.5) - 0.5*log(det(C1)) - 0.5*((xt-m1')*inv(C1)*(x-m1));
d2 = log(0.5) - 0.5*log(det(C2)) - 0.5*((xt-m2')*inv(C2)*(x-m2));
d21 = d2 - d1;
vpa(simplify(d21),2)
end
