%% Pattern Recognition - Seminar 2
% Author: Diego Rodriguez, 174458, u114355
% Practical Exercises:

%% Exercise 1

% Load the datasets “ionosphere”, and compare the performance of LDA and
% Support Vector machines on the dataset. To do so, you will have to divide
% the dataset in a training set and a test set, train your models with the
% training set, and apply them to the test set to compute the accuracy of
% each model.

% Load the dataset
disp("Loading dataset...");
drawnow;
load ionosphere;

%disp("Removing the second feature of the ionosphere dataset... var = 0");
%X = X(:, [1, 3:end]);

clc;
% Separate the dataset into training and test sets
disp("Separating into training and test set.");
drawnow;
p = .7; % proportion of rows to select for training
N = size(X,1); % total number of rows
tf = false(N,1); % create logical index vector
tf(1:round(p*N)) = true;
tf = tf(randperm(N)); % randomise order
Xtrain = X(tf,:);
Ytrain = Y(tf,:);
Xtest = X(~tf,:);
Ytest = Y(~tf,:);

% Train the LDA Model using the Training Set.
disp("Training LDA Model");
drawnow;
ldaModel=fitcdiscr(Xtrain,Ytrain, 'DiscrimType', 'pseudolinear'); %LDA

% Train the SVM Model using the Training Set.
disp("Training SVM Model");
drawnow;
svmModel = fitcsvm(Xtrain, Ytrain);
% Predict the test values
% LDA
ldaPrediction=predict(ldaModel, Xtest);
lda_accu = mean(strcmp(ldaPrediction, Ytest));
disp("The LDA Accuracy on the test set is: "+lda_accu);

% SVM

svmPrediction = predict(svmModel, Xtest);
svm_accu =mean(strcmp(Ytest,svmPrediction));
disp("The SVM Accuracy on the test set is: "+svm_accu);

%% Exercise 1 using a 10 fold cross validation.
clc;

load ionosphere;

disp("Using 10-Fold Cross Validation...");

% Train the LDA Model using the Training Set.
disp("Training LDA Model");
drawnow;
ldaModel=fitcdiscr(Xtrain,Ytrain, 'DiscrimType', 'pseudolinear', 'crossVal', 'on'); %LDA

% Train the SVM Model using the Training Set.
disp("Training SVM Model");
drawnow;
svmModel = fitcsvm(Xtrain, Ytrain, 'crossVal', 'on');

% Predict the test values
% LDA
lda_accu = 1 - kfoldLoss(ldaModel);
disp("The LDA Accuracy on the test set is: "+lda_accu);

% SVM

lda_accu = 1 - kfoldLoss(svmModel);
disp("The SVM Accuracy on the test set is: "+lda_accu);

%% Exercise 2: PLS vs PCR

% Imagine that you are hired for an aeroline to find an easy and cheap way
% to determine the octane ratio of the gasoline (which determines the
% gasoline quality) without testing it in real planes. They give you 60
% samples of gasoline at 401 wavelengths, and their octane ratings.
clf;
% Load the dataset and visualize it
clc; clearvars;
disp("Loading dataset...");
drawnow;
load spectra
X=NIR;
y=octane;
[dummy,h] = sort(octane);
oldorder = get(gcf,'DefaultAxesColorOrder');
set(gcf,'DefaultAxesColorOrder',jet(60));
plot3(repmat(1:401,60,1)',repmat(octane(h),1,401)',NIR(h,:)');
set(gcf,'DefaultAxesColorOrder',oldorder);
xlabel('Wavelength Index'); ylabel('Octane'); axis('tight');
grid on

% Separate

disp("Separating into training and test set...");
drawnow;
p = .7; % proportion of rows to select for training
N = size(X,1); % total number of rows
tf = false(N,1); % create logical index vector
tf(1:round(p*N)) = true;
tf = tf(randperm(N)); % randomise order
Xtrain = X(tf,:);
Ytrain = y(tf,:);
Xtest = X(~tf,:);
Ytest = y(~tf,:);

%% Section A:

% Build three models using Multivariate Linear Regression, PLS and PCR to
% predict the octane ratio of the gasoline given its wavelengths.
disp("Calculating regression using MVR, PCR and PLS");
% Multi-Variate Single Regression

betaMR = regress(Ytrain-mean(Ytrain),Xtrain); % Multi Var. Regression 1-Predictor Var.
% Where betaMR are the coefficients in the linear regression for each of
% the wavelengths spectral response.

% PLS
N = 10; % Number of dimensions

% [... , betaPLS] = plsregress(X,Y,ndims)  %PLS
% Using Cross Validation and 10 Dimensions
[xl,yl,xs,ys,betaPLS,pctvar,mse] = plsregress(Xtrain,Ytrain,N);
% plot(0:10,mse(2,:),'-o');
% octaneFitted = [ones(size(NIR,1),1) NIR]*betaPLS;
% plot(octane,octaneFitted,'o');

% PCR

[PCALoadings,PCAScores,PCAVar] = pca(Xtrain,'Economy',false);
betaPCR = regress(Ytrain-mean(Ytrain), PCAScores(:,1:2));
betaPCR = PCALoadings(:,1:2)*betaPCR;
betaPCR = [mean(Ytrain) - mean(Xtrain)*betaPCR; betaPCR];

% Prediction

yPredMR= Xtest*betaMR + mean(Ytrain);
yPredPCR = [ones(size(Xtest, 1), 1) Xtest]*betaPCR;
yPredPLS=[ones(size(Xtest, 1), 1), Xtest]*betaPLS;


yTrainPredMR= Xtrain*betaMR + mean(Ytrain);
yTrainPredPCR = [ones(size(Xtrain, 1), 1) Xtrain]*betaPCR;
yTrainPredPLS=[ones(size(Xtrain, 1), 1), Xtrain]*betaPLS;

yError = (abs([Ytest - yPredMR, Ytest - yPredPCR, Ytest - yPredPLS]));
yTrainError = (abs([Ytrain - yTrainPredMR, Ytrain - yTrainPredPCR, Ytrain - yTrainPredPLS]));
meanYError = mean(yError);
meanYTrainError = mean(yTrainError);
models = {"MVR", "PCR", "PLS"};
%% Section B:
clf;
figure(1);
subplot(2, 1, 1);
hold on; 
errorbar(yPredMR, yError(:, 1),'o','MarkerSize',5,...
'MarkerEdgeColor','red','MarkerFaceColor','red', 'Color', 'Red')

errorbar(yPredPCR, yError(:, 2),'o','MarkerSize',5,...
'MarkerEdgeColor','green','MarkerFaceColor','green', 'Color', 'green');

errorbar(yPredPLS, yError(:, 3),'o','MarkerSize',5,...
'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'Color', 'blue')

plot(Ytest, 'x', 'markerSize', 14, 'lineWidth', 2);

legend("MVR", "PCR", "PLS", "Test Samples"); hold off;
title("Sample vs Octane Ratio Prediction");
xlabel("Sample Number");
ylabel("Octane Ratio");
grid on;

subplot(2,1,2);
[~, idx] = min(meanYError);
yPred = [yPredMR, yPredPCR, yPredPLS];
hold on;
errorbar(yPred(:, idx), yError(:, idx),'o','MarkerSize',5,...
'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'Color', 'blue')
plot(Ytest, 'x', 'markerSize', 14, 'lineWidth', 1);
legend("Regression", "Test Samples");
title("Best Fitted Regression: "+models(idx));
xlabel("Sample Number");
ylabel("Octane Ratio");
grid on;
hold off;

% Training Examples
hold off;
input('Press Enter to plot the training samples.\n');
clf;
figure(1);
subplot(2, 1, 1);
hold on; 
errorbar(yTrainPredMR, yTrainError(:, 1),'o','MarkerSize',5,...
'MarkerEdgeColor','red','MarkerFaceColor','red', 'Color', 'Red')

errorbar(yTrainPredPCR, yTrainError(:, 2),'o','MarkerSize',5,...
'MarkerEdgeColor','green','MarkerFaceColor','green', 'Color', 'green');

errorbar(yTrainPredPLS, yTrainError(:, 3),'o','MarkerSize',5,...
'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'Color', 'blue')

plot(Ytrain, 'x', 'markerSize', 14, 'lineWidth', 2);

legend("MVR", "PCR", "PLS", "Test Samples"); hold off;
title("Sample vs Octane Ratio Prediction");
xlabel("Sample Number");
ylabel("Octane Ratio");
grid on;

subplot(2,1,2);
[~, idx] = min(meanYTrainError);
yPred = [yTrainPredMR, yTrainPredPCR, yTrainPredPLS];
hold on;
errorbar(yPred(:, idx), yTrainError(:, idx),'o','MarkerSize',5,...
'MarkerEdgeColor','blue','MarkerFaceColor','blue', 'Color', 'blue')
plot(Ytrain, 'x', 'markerSize', 14, 'lineWidth', 1);
legend("Regression", "Test Samples");
title("Best Fitted Regression: "+models(idx));
xlabel("Sample Number");
ylabel("Octane Ratio");
grid on;
hold off;

%% Section C:
disp("Mean Absolute Error (Test Set / Validation Set):");
disp(models);
disp(meanYError);

%% Section D:

%  After that build again a PLS and a PCR but using 10 components in both
%  cases and make a plot of the variance as a function of the number of components (from 1 to
% 10).

clf;
% Load the dataset and visualize it
clc; clearvars;
disp("Loading dataset...");
drawnow;
load spectra
X=NIR;
y=octane;
% Begin the variance plot
Ndim = 10;
PLSbuffer = zeros(Ndim, 1);
PCRbuffer = PLSbuffer;
for Ndim = 1:10
    % Separate

    disp("Separating into training and test set...");
    drawnow;
    p = .7; % proportion of rows to select for training
    N = size(X,1); % total number of rows
    tf = false(N,1); % create logical index vector
    tf(1:round(p*N)) = true;
    tf = tf(randperm(N)); % randomise order
    Xtrain = X(tf,:);
    Ytrain = y(tf,:);
    Xtest = X(~tf,:);
    Ytest = y(~tf,:);

    disp("Calculating PLS...");
    drawnow;
    % PLS

    % [... , betaPLS] = plsregress(X,Y,ndims)  %PLS
    % Using Cross Validation and 10 Dimensions
    [xl,yl,xs,ys,betaPLS,pctvar,mse] = plsregress(Xtrain,Ytrain,Ndim);
    % plot(0:10,mse(2,:),'-o');
    % octaneFitted = [ones(size(NIR,1),1) NIR]*betaPLS;
    % plot(octane,octaneFitted,'o');

    % PCR
    disp("Calculating PCR...");
    drawnow;
    [PCALoadings,PCAScores,PCAVar] = pca(Xtrain,'Economy',false);
    betaPCR = regress(Ytrain-mean(Ytrain), PCAScores(:,1:Ndim));
    betaPCR = PCALoadings(:,1:Ndim)*betaPCR;
    betaPCR = [mean(Ytrain) - mean(Xtrain)*betaPCR; betaPCR];

    disp("Predicting Results...");

    % Prediction
    % Test Set
    yPredPCR = [ones(size(Xtest, 1), 1) Xtest]*betaPCR;
    yPredPLS=[ones(size(Xtest, 1), 1), Xtest]*betaPLS;

    % Training Set
    yTrainPredPCR = [ones(size(Xtrain, 1), 1) Xtrain]*betaPCR;
    yTrainPredPLS=[ones(size(Xtrain, 1), 1), Xtrain]*betaPLS;

    yError = (abs([Ytest - yPredPCR, Ytest - yPredPLS]));
    yTrainError = (abs([Ytrain - yTrainPredPCR, Ytrain - yTrainPredPLS]));
    meanYError = mean(yError);
    meanYTrainError = mean(yTrainError);
    models = {"PCR", "PLS"};
    disp("Mean Absolute Error (Test Set / Validation Set):");
    disp(models);
    disp(meanYError);
    disp("Variance [PCR]: "+ var(yPredPCR));
    disp("Variance [PLS]: "+ var(yPredPLS));
    PCRbuffer(Ndim) = var(yPredPCR);
    PLSbuffer(Ndim) = var(yPredPLS);
end


plot(1:Ndim, PCRbuffer, 'r-'), hold on;
plot(1:Ndim, PLSbuffer, 'b-');
xlabel("Number of Components");
ylabel("\sigma^2");
title("\sigma^2 vs N of Dim.");
legend("PCR", "PLS");
%% Section E:

% e) Finally, make a plot to show how does the error in the predictions
% depend on the number of components used. To do so, you will have to use
% cross-vaildation to determine the error produced after applying PLS and
% PCR with 1,2...10 components.

clf;
% Load the dataset and visualize it
clc; clearvars;
disp("Loading dataset...");
drawnow;
load spectra
X=NIR;
y=octane;
[n, p] = size(X);

PCRAbsErr = zeros(10, 1);
PLSAbsErr = PCRAbsErr;       
    % PCR
[PCALoadings,PCAScores,PCAVar] = pca(X,'Economy',false);
betaPCR = regress(y-mean(y), PCAScores(:,1:Ndim));
betaPCR = PCALoadings(:,1:Ndim)*betaPCR;
betaPCR = [mean(y) - mean(X)*betaPCR; betaPCR];
PCRAbsErr(Ndim) = mean(abs([ones(size(X, 1), 1) X]*betaPCR - y));

% PLS
[Xl,Yl,Xs,Ys,betaPLS,pctVar,PLSmsep] = plsregress(X,y,Ndim,'CV',10);

PLSabsErr(Ndim) = mean(abs([ones(size(X, 1), 1) X]*betaPLS - y));
end

% Plot the Mean

plot(1:10,PLSabsErr,'b-o',1:10,PCRAbsErr,'r-^');
xlabel('Number of components');
ylabel('Estimated Absolute Prediction Error');
legend({'PLSR' 'PCR'},'location','NE');
title("$$ \epsilon = \frac{1}{n}|y - \hat{y}| $$", "interpreter", "latex");

% References:

% https://es.mathworks.com/help/stats/examples/partial-least-squares-regression-and-principal-components-regression.html#d119e5749
% https://es.mathworks.com/help/stats/plsregress.html