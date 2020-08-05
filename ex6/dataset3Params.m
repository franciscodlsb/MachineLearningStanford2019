function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [.01 .03 .1 .3 1 3 10 30];
sigma = [.01 .03 .1 .3 1 3 10 30];

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

count = 0;

for ii = 1:length(C)
    for jj = 1:length(sigma)
count = count+1;
model= svmTrain(X, y, C(ii), @(x1, x2) gaussianKernel(x1, x2, sigma(jj)));         
predictions = svmPredict(model, Xval);
error(count) = mean(double(predictions ~= yval));
iijj(count,1) = ii;
iijj(count,2) = jj;

    end
end

[~, best] = min(error);
C = C(iijj(best,1));
sigma = sigma(iijj(best,2));

% =========================================================================

end
