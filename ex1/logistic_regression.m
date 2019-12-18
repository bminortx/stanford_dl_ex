function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  n=size(X,1);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  
  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
  
  for j = 1 : m
      meas_val = 1 / (1 + exp(-theta' * X(:, j)));
      cost_one = y(j) * log(meas_val);
      cost_zero = (1 - y(j)) * log(1 - meas_val);
      f = f - (cost_one + cost_zero);
      for i = 1 : n
          g(i) = g(i) + (X(i, j) * (meas_val - y(j)));
      end
  end
