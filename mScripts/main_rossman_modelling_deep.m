%% apply generalized logit transform
target_ = log(( ntrain.Sales +1000)./ (max(ntrain.Sales+1000)-ntrain.Sales ));
% min-max normalisation to [0.2-0.8]
sum(isinf(target_))
sum(isnan(target_))
hist(target_,100)
%%
target = (target_ - min(target_)) / (max(target_)-min(target_)) * 0.2 + 0.5;
sum(isinf(target))
sum(isnan(target))

hist(target,100)

%% deep NN
hiddenSize1 = 13;
autoenc1 = trainAutoencoder(mat(Indices==1,:)',hiddenSize1, ...
  'MaxEpochs',50, ...
  'L2WeightRegularization',0.004, ...
  'SparsityRegularization',4, ...
  'SparsityProportion',0.15, ...
  'ScaleData', false);

%%
feat1 = encode(autoenc1,mat(Indices==1,:)');
%% train the second layer
hiddenSize2 = 5;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
  'MaxEpochs',100, ...
  'L2WeightRegularization',0.002, ...
  'SparsityRegularization',4, ...
  'SparsityProportion',0.1, ...
  'ScaleData', false);
%%
feat2 = encode(autoenc2,feat1);
%%
softnet = trainSoftmaxLayer(feat2,target(Indices==1)','MaxEpochs',400,'LossFunction','mse');

% hiddenSize3=1;
% autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
%   'MaxEpochs',50, ...
%   'L2WeightRegularization',0.002, ...
%   'SparsityRegularization',4, ...
%   'SparsityProportion',0.1, ...
%   'ScaleData', false);
%%
deepnet = stack(autoenc1,autoenc2,softnet);

view(deepnet)
%%

[deepnet, tr]= train(deepnet,mat(Indices==2,:)',target(Indices==2)');
%%
ypred = deepnet(mat(Indices==3,:)');
rsquared(target(Indices==3), ypred',1)
%%
feat1 = encode(autoenc1,mat(Indices==2,:)');
feat2 = encode(autoenc2,feat1);
X=[ones(sum(Indices==2),1) feat2'];
b = X\target(Indices==2);
ypred=X*b;
figure(1);clf;
rsquared(target(Indices==2), ypred, 1);

%%
feat1 = encode(autoenc1,mat(Indices==2,:)');
X=[ones(sum(Indices==2),1) feat1'];
b = X\target(Indices==2);
ypred=X*b;
figure(2);clf;
rsquared(target(Indices==2), ypred, 1);

