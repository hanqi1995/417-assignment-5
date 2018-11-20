load zip.train;


subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y = subsample(:,1);
X = subsample(:,2:257);

Mdl1 = BaggedTreesMdl(X, Y, 1);

Mdl200 = BaggedTreesMdl(X, Y, 200);

load zip.test;
subsampletest = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Ytest = subsampletest(:,1);
Xtest = subsampletest(:,2:257);


error1 = mean(predict(Mdl1{1}, Xtest)~=Ytest);
fprintf('The test error of 1 tree is %.4f\n', error1);
t = zeros(size(Xtest,1),1);
for i = 1:200
    t = t + (predict(Mdl200{i}, Xtest) == Ytest)*2-1;
end

error200 = -mean((sign(t) - 1)/2);
fprintf('The test error of 200 trees is %.4f\n', error200);

load zip.train;


subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y = subsample(:,1);
X = subsample(:,2:257);

Mdl1 = BaggedTreesMdl(X, Y, 1);

Mdl200 = BaggedTreesMdl(X, Y, 200);

load zip.test;
subsampletest = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Ytest = subsampletest(:,1);
Xtest = subsampletest(:,2:257);


error1 = mean(predict(Mdl1{1}, Xtest)~=Ytest);
fprintf('The test error of 1 tree is %.4f\n', error1);
t = zeros(size(Xtest,1),1);
for i = 1:200
    t = t + (predict(Mdl200{i}, Xtest) == Ytest)*2-1;
end

error200 = -mean((sign(t) - 1)/2);
fprintf('The test error of 200 trees is %.4f\n', error200);

