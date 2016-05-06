%%
load main_rossman_explore.mat ntrain ntest

%%
plot(ntrain.Customers, ntrain.Sales,'.');
xlabel('Customer');
ylabel('Sales');
%print('-dpng', 'Pictures/main_rossman_modelling__customer_vs_sales.png');
%%
clc
ntrain.Properties.VariableNames'
ntest.Properties.VariableNames'

%%
grpstats(ntrain, {'Promo','SchoolHoliday','StateHoliday','StoreType','Open'}, {'mean','median'} ,'DataVars',{'Sales'})
%% filter out days on which the shop is closed
ntrain(ntrain.Open==0,:) =[];

%%
unique(ntrain.DayOfWeek)
ntrain.DayOfWeek = categorical(ntrain.DayOfWeek);
ntest.DayOfWeek = categorical(ntest.DayOfWeek);

%%
%grpstats(ntrain, {'Promo','SchoolHoliday','StateHoliday','StoreType','Open'}, {'mean','median'} ,'DataVars',{'Sales'})
grpstats(ntrain, {'Promo','Promo2', 'SchoolHoliday','StateHoliday','StoreType','Assortment'}, {'mean','median'} ,'DataVars',{'Sales'})

%%
grpstats(ntrain, {'Promo','SchoolHoliday','StateHoliday','DayOfWeek'}, {'mean','median'} ,'DataVars',{'Sales'})

%%
plot(ntrain.CompetitionDistance, ntrain.Sales,'.');
xlabel('Competition Distance');
ylabel('Sales');
print('-dpng', 'Pictures/main_rossman_modelling__competition_dist_vs_sales.png');
%% Using regression tree
clc
ntrain.Properties.VariableNames'
selected_var = [2 6:18];
%selected_var = [2 5 6:18];
target_var = 4;
{ntrain.Properties.VariableNames{selected_var}}'

%% Generate training, validation and test sets
rng('default');
Indices = crossvalind('Kfold', size(ntrain, 1), 3);

%%
clear Mdl;

%Mdl.linear = fitlm(ntrain(:, [selected_var, target_var]));

%% LINEAR model: Good prediction when the customer size is known
selected_var = [2 5 6:18]; %include Sales
target_var = 4;
Mdl.linear_robust = fitlm(ntrain(Indices==1, [selected_var, target_var]),'RobustOpts','on');

ypred = predict(Mdl.linear_robust,ntrain(Indices==3, [selected_var, target_var]));

plot(ntrain.Sales(Indices==3), ypred,'.');
xlabel('Actual Sales');
ylabel('Predicted Sales');

rsquared(ntrain.Sales(Indices==3), ypred)

%print('-dpng', 'Pictures/main_rossman_modelling__linear_robust_mdl_w_customer.png');
%% Bad prediction when the customer size is not known
selected_var = [2 6:18]; %excluse Sales
target_var = 4;
Mdl.linear_robust = fitlm(ntrain(Indices==1, [selected_var, target_var]),'RobustOpts','on');

ypred = predict(Mdl.linear_robust,ntrain(Indices==3, [selected_var, target_var]));

selected = (isnan(ypred));

figure(2);
plot(ntrain.Sales(Indices==3), ypred,'.');
xlabel('Actual Sales');
ylabel('Predicted Sales');

rsquared(ntrain.Sales(Indices==3), ypred)


%print('-dpng', 'Pictures/main_rossman_modelling__linear_robust_mdl_wo_customer.png');

%% Need to check where NaN came from
map = zeros(size(ntrain));
for i=1:numel(ntrain.Properties.VariableNames),
  cmd_ = sprintf('tmp = ntrain.%s;', ntrain.Properties.VariableNames{i});
  eval(cmd_);
  if isnumeric(tmp),
    map(:,i)=isnan(tmp);
  end;
end
%% now regression tree

selected_var = [2 6:18]; %excluse Sales
Mdl.tree = fitrtree(ntrain(Indices==1, selected_var), ntrain.Sales(Indices==1)); %,'Kfold', 10);

%% prune the tree
mList = 0:1000:round(max(Mdl.tree.PruneList));
rs=zeros(numel(mList),1);
for m=1:numel(mList), 
  tree_ = prune(Mdl.tree,'Level', mList(m));
  ypred = predict(tree_,ntrain(Indices==2, selected_var));
  rs(m) = rsquared(ntrain.Sales(Indices==2), ypred); % mean( abs(ntrain.Sales(Indices==2) - ypred));
  fprintf(1,'.');
end;

%% prune and save
close all;
bar(mList, rs);
ylabel('R-squared');
xlabel('Prune level');
%%
print('-dpng', 'Pictures/main_rossman_modelling__rtree_pruned_level.png');
[~, index] = max(rs);
%%
Mdl.ptree = prune(Mdl.tree,'Level', mList(index));

%% testing
selected_var = [2 6:18]; %excluse Sales
ypred = predict(Mdl.ptree, ntrain(Indices==3, selected_var));
%%
rsquared(ntrain.Sales(Indices==3), ypred, 1);
print('-dpng', 'Pictures/main_rossman_modelling__rtree_pruned_scatter.png');
%
%%
rsquared(ntrain.Sales(Indices==3), ypred)
%%
clc
view (Mdl.ptree)
%%
view(Mdl,'mode','graph'); 

%% Represent the data differently neural network
clear u_;
selected_var = [2 7:18]; %excluse Sales and Open

for i=1:numel(selected_var),
  u_{i}=unique(ntrain(Indices==1, selected_var(i)));
end;
%% Deal with the undefined value of PromoInterval 
[dat_, label_] = findgroups(ntrain.PromoInterval);
dat_(isnan(dat_)) = 4;
label_ = categories(label_);
label_ = {label_{:}, 'Undefined'};
ntrain.PromoInterval = categorical(dat_);
PromoInterval_label = label_;

%% Transform all variables to dummyvar except a few
selected_var = [2 7:18]; %excluse Sales and Open

tmp= ntrain.Properties.VariableNames;
{tmp{selected_var}}'

list = cellfun(@(x) numel(x), u_);
%% translate the data into dummyvar
mat=[];
for i=1:numel(selected_var),
  if list(i)>30,
    tmp = table2array(ntrain(:, selected_var(i)));
    mat=[mat zscore(tmp)];
  else
    [dat_,groups_] = findgroups(ntrain(:, selected_var(i)));
    tmp = dummyvar(dat_);
    mat=[mat tmp];
  end;
end;
%%
imagesc(mat);

%% PLSR
[XL,yl,XS,YS,beta,PCTVAR] = plsregress(mat(Indices==1,:), ntrain.Sales(Indices==1),20);
%%
plot(1:20,cumsum(100*PCTVAR(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');
%%
[XL,yl,XS,YS,beta,PCTVAR] = plsregress(mat(Indices==1,:), ntrain.Sales(Indices==1),12);
Mdl.plsr_beta = beta;
%%
yfit = [ones(sum(Indices==3),1) mat(Indices==3,:)]*Mdl.plsr_beta;
plot(ntrain.Sales(Indices==3), yfit,'.');
xlabel('Actual Sales');
ylabel('Predicted Sales');
%%
print('-dpng', 'Pictures/main_rossman_modelling__plsr.png');
rsquared(ntrain.Sales(Indices==3), yfit);
%%
%%
Mdl.tree = fitrtree(mat(Indices==1, :), ntrain.Sales(Indices==1)); %,'Kfold', 10);

%Mdl.plsr_dummyvar = fitlm([ mat(Indices==1, :), ntrain.Sales ]),'RobustOpts','on');

%% Try a feedforward NN
net = feedforwardnet(10);
net = train(net,mat(Indices==1,:)',ntrain.Sales(Indices==1)');
%%
ypred = net(mat(Indices==3,:)');
%%
rsquared(ntrain.Sales(Indices==3), ypred', 1);
%%
print('-dpng', 'Pictures/main_rossman_modelling__mlp.png');

%% Do fusion here
ypred1 = net(mat(Indices==2,:)');
selected_var = [2 6:18]; %excluse Sales
ypred2 = predict(Mdl.ptree, ntrain(Indices==2, selected_var));
%%
plot(ypred1, ypred2,'.');