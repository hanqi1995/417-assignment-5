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
obs = size(X,1);
n = size(X,2);
Db = zeros(obs,n,numBags);
t = zeros(obs,1);
Mdl = cell(n);
for i = 1:numBags
    Db(:,:,i) = datasample(X, obs);
    Mdl{i} = fitctree(Db(:,:,i), Y);
    xi = ~ismember(X, Db(:,:,i), 'rows');
    for j = 1:obs
        if(xi(j))
             if (predict(Mdl{i}, X(j,:)) == Y(j))
                t(j) = t(j)+1;
            else
                t(j) = t(j)-1;
             end
        end
    end
end
%break the ties
%{
for i = 1:obs
    if(t(i) == 0)
        t(i) = sign(rand-0.5);
    end
end
%}
oobErr = -mean((sign(t) - 1)/2);
end
