function [ oobErr ] = BaggedTrees( X, Y, numBags )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function
n = size(X,1);
error = zeros(n,1);
%...
for i = 1:numBags
    for j = 1:n
        numError = 0;
        data = X;
        data(j) = [];
        Db = datasample(data, n);
        Mdl = fitctree(Db, Y);
        if(predict(Mdl, X(j))~=Y(j))
            numError = numError + 1;
        end
    end
    error(i) = numError / n;
end

oobErr = mean(error);

end
