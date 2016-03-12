function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0; % commented by Saeid
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hOfX = sigmoid(X * theta);
cost = -(y .* log(hOfX) + (ones(m, 1) - y) .* log(ones(m, 1) - hOfX));
J = sum(cost) / m;

n = length(theta);
regularizer = theta .^ 2;
J = J + (lambda / (2 * m)) * sum(regularizer(2:n));

hOfX = hOfX - y; % hOfX is Error!
grad = X' * hOfX / m;
grad(2:n) = grad(2:n) + (lambda / m) * theta(2:n);




% =============================================================

end
