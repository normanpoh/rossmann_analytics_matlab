%% Check the documentation here
% https://www.kaggle.com/c/rossmann-store-sales/data?train.csv.zip

%% load data
%store = importfile_data('../data/store.csv');
ds = datastore('../data/store.csv');
ds.MissingValue = 0; % Strategy for missing value
store = read(ds);




%%
store.Assortment = categorical(store.Assortment);
store.StoreType = categorical(store.StoreType);

%count_ = cellfun(@(x) numel(x), store.PromoInterval);
store.PromoInterval = categorical(store.PromoInterval);

%% load the test set and see what variables are valid and can be used for prediction
dat_test = datastore('../data/test.csv');
dat_test.ReadSize = 41089-1 ; 
test = read(dat_test);  

%test = importfile_test('../data/test.csv'); %can't use this function

%% load using the training set
dat = datastore('../data/train.csv');
dat.MissingValue = 0;% Strategy for missing value

preview(dat)

% check the data store
dat.VariableNames
dat.TextscanFormats

%dat = importfile_data('../data/train.csv'); % file too big to be loaded: > 1 million rows
N = 1017210-1; %100000; unix('wc -l train.csv')
reset(dat);
dat.ReadSize = round(N) ; 
T = read(dat);

%% variables that can be used for the prediction purpose
clc
test.Properties.VariableNames'

T.Properties.VariableNames'

%% Check variable relationships
plot(T.Customers, T.Sales,'.');
xlabel('Number of customers');
ylabel('Sales amount');

%%
corrcoef(T.Sales, T.Customers)
%% check if the data was loaded correctly
unique(test.StateHoliday)
unique(T.StateHoliday)


%%
clc
T.Properties.VariableNames
test.Properties.VariableNames

%%
summary(test)


%% join the table based on store

ntest = join(test,store,'Keys','Store');
ntrain = join(T,store,'Keys','Store');

clear T test dat*


%% conver the variable types
unique(ntrain.StateHoliday)
ntrain.StateHoliday = categorical(ntrain.StateHoliday);
ntest.StateHoliday = categorical(ntest.StateHoliday);

%%
unique(ntest.SchoolHoliday)
ntest.SchoolHoliday = str2double( ntest.SchoolHoliday );
ntrain.SchoolHoliday = str2double( ntrain.SchoolHoliday );

%%
save main_rossman_explore.mat ntrain ntest
%% STOP HERE
%% Read the rest of the file
while hasdata(dat)
    T = read(dat);
end

