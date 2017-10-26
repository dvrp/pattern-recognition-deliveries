clearvars;
format short;
disp('Reading samples...');
samples = [4.9, 3.2, 1.7, 0.2; 5, 3.2, 1.6, 0.5; 5.5, 2.8, 3.6, 1.3; 7.1, 3.1, 6.1, 1.7]; % Imaginary Samples

% Labels
% 1: Setosa
% 2: Versicolor
% 3: Virginica

addpath('../dataset');
dataset = csvread('data.csv');
labels = ["setosa", "versicolor", "virginica"];
% Clean the dataset

sepal_length = dataset(:, 1);
petal_length = dataset(:, 2);
sepal_width  = dataset(:, 3);
petal_width  = dataset(:, 4);
classId      = dataset(:, 5);
X            = [sepal_length, petal_length, sepal_width, petal_width]; % Features
y            = classId;                                                % Variable we want to predict

disp('Applying PCA...');
%% Apply PCA to matrix X before extraction of features for each class
%====================================================================

% Generating covariance matrix
X_cov = cov(X);

% Generating the means for each feature
X_mu = mean(X, 1);

%% Extracting eigens
% U: Eigenvectors
% V: Eigenvalues
% Find the highest eigenvalues using Matlab's implementation of the singular value decomposition function
%[U, V, W] = svd(X_cov)
% For learning purposes, I decided to use eig
[U, V] = eig(X_cov);

% Find the highest N features.
NUMBER_FEATURES = 2;
V = sum(V, 2); % Transform matrix into vector of eigenvalues
totalVar = sum(V);
[V, idx] = sort(V, 'descend'); % Sort the matrix
disp(['Using ' num2str(NUMBER_FEATURES) ' out of ' num2str(size(X, 2)) ' features...']);
V = V(1:NUMBER_FEATURES);
pickedVar= sum(V); % Store the variability
disp([num2str(100*pickedVar/totalVar), '% variability retained']);
U = U(:, idx(1:NUMBER_FEATURES));

%% Reproject the data onto X_proj
% Find the values projection on the new basis
T = X * U;

% Dimension-reduced data: X_proj
X_proj = T * U';

% PCA Finished! -> New dataset -> T
%====================================================================

% Extract features from each class

X_setosa =     T(find(y(:) == 1), :);
X_versicolor = T(find(y(:) == 2), :);
X_virginica =  T(find(y(:) == 3), :);

% Calculate cov. matrices

setosa_covMat = cov(X_setosa);
versicolor_covMat = cov(X_versicolor);
virginica_covMat = cov(X_virginica);

% Extract the means

setosa_mean = mean(X_setosa, 1);
versicolor_mean = mean(X_versicolor, 1);
virginica_mean = mean(X_virginica, 1);

% Calculate the pdf of the samples
% But convert samples to new space before
samples = samples * U;

setosa_pdf = mvnpdf(samples, setosa_mean, setosa_covMat);
versicolor_pdf =  mvnpdf(samples, versicolor_mean, versicolor_covMat);
virginica_pdf =  mvnpdf(samples, virginica_mean, virginica_covMat);

% Probability of the samples being setosa
prob_setosa = (setosa_pdf)./(setosa_pdf + versicolor_pdf + virginica_pdf);

% Probability of the samples being setosa
prob_versicolor = (versicolor_pdf)./(setosa_pdf + versicolor_pdf + virginica_pdf);

% Probability of the samples being setosa
prob_virginica = (virginica_pdf)./(setosa_pdf + versicolor_pdf + virginica_pdf);

matProbabilities = [prob_setosa, prob_versicolor, prob_virginica];

for i=1:size(matProbabilities, 1)
    [value, idx] = max(matProbabilities(i, :));
    disp(['The sample ', num2str(i), ' is a ', char(labels(idx)), '. Confidence: ', num2str(value*100), ' %']);
end