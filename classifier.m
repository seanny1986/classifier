clear all;
close all;

% import neural network functions
addpath(genpath('/home/seanny/Dropbox/Documents/Octave Scripts/blaze'))

tr = csvread('train.csv', 1, 0);                  % read train.csv
sub = csvread('test.csv', 1, 0);                  % read test.csv

figure                                          % plot images
colormap(gray)                                  % set to grayscale
for i = 1:25                                    % preview first 25 samples
    subplot(5,5,i)                              % plot them in 6 x 6 grid
    digit = reshape(tr(i, 2:end), [28,28])';    % row = 28 x 28 image
    imagesc(digit)                              % show the image
    title(num2str(tr(i, 1)))                    % show the label
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);
end

% pull out data
DATA = tr(:,2:end);
LABEL = tr(:,1);

% put data into the shape we want it (each example is a column vector)
DATA = DATA';
LABEL = LABEL';
TEST = sub';

% map data labels to values 0 and 1
CLASSIFICATION = zeros(10,size(LABEL,2));
for i=1:size(LABEL,2),
  CLASSIFICATION(LABEL(i)+1,i) = 1;
end

% create neural network
TOPOLOGY = [784 50 10];
[THETAs Xs] = nnbuild(TOPOLOGY);
ACTFNS = cell(1,size(TOPOLOGY,2)-1);
ACTFNS(1:end) = @sigmoid;
options = optimset('MaxIter',1000);

% train the network on the dataset
THETAs = nntrain(THETAs,Xs,DATA,CLASSIFICATION,TOPOLOGY,ACTFNS,@crssenterr,@fmincg,0,options);
OUT = cell2mat(nnfeedforward(THETAs,Xs,TEST,ACTFNS)(end));
TEST = TEST';

% convert prediction back into its integer value
PREDICTION = zeros(size(OUT,2),1);
for i=1:size(OUT,2),
  PREDICTION(i) = find(OUT(:,i)==max(OUT(:,i)))-1;
end

% prediction on test set for first 25 samples
figure    
colormap(gray)                   
for i = 1:25                                    
    subplot(5,5,i)                              
    digit = reshape(TEST(i, 1:end), [28,28])';  
    imagesc(digit)                              
    title(num2str(PREDICTION(i, 1)))      
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);    
end
