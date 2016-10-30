function [ d21 ] = GaussianBayes( n, t1, t2 )
% Bayes classifier for Gaussian pattern classes
% - use training data to form the decision function, assumed two training
% data have the same amount.
% input:
% 	n: the amount of features (dimensions)
%	t1: training data of class 1
%	t2: training data of class 2

% Adjust the size of training data
[~, col1] = size(t1);
[~, col2] = size(t2);
if(col1 == n)
    t1 = t1';
end
if(col2 == n)
    t2 = t2';
end
[~, N] = size(t1);

% Create the covariance matrices
C1 = zeros(n); C2 = zeros(n);
for i = 1:N
    C1 = C1 + t1(:,i) * t1(:,i)';
    C2 = C2 + t2(:,i) * t2(:,i)';
end
m1 = mean(t1,2); m2 = mean(t2,2);
C1 = C1/N - m1 * m1';
C2 = C2/N - m2 * m2';

% Calculate the decision function and surface
x = sym('x', [n 1], 'rational');
if(det(C1)==0)
    dC1 = 1;
else
    dC1 = det(C1);
end
if(det(C2)==0)
    dC2 = 1;
else
    dC2 = det(C2);
end
d1 = log(0.5) - 0.5*log(dC1) - 0.5*((x-m1)'*pinv(C1)*(x-m1));
d2 = log(0.5) - 0.5*log(dC2) - 0.5*((x-m2)'*pinv(C2)*(x-m2));
d21 = vpa(simplify(d2 - d1),2);
end
