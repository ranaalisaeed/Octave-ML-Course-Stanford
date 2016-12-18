function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

steps = [0.01 0.03 0.1 0.3 1 3 10 30];
err_min = inf;

fprintf('starting to go through C and sigma steps to find optimals\n');
for C = steps
    for sigma = steps
        fprintf('trying C, sigma = [%f %f]\n',C,sigma);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        fprintf('error = %f\n',err);
        if err < err_min
            err_min = err;
            C_opt = C;
            sigma_opt = sigma;
            fprintf('found new min, C = %f, sigma = %f, with error = %f\n',C_opt,sigma_opt,err_min);
        endif
    end
end

C = C_opt;
sigma = sigma_opt;

fprintf('Optimal values C, sigma = [%f %f] with prediction error = %f\n',C,sigma,err_min);

% =========================================================================

end
