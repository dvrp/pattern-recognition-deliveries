clearvars;
format short;
disp('Reading samples...');
samples = [4.9, 3.2, 1.7, 0.2; 5, 3.2, 1.6, 0.5; 5.5, 2.8, 3.6, 1.3; ...
    7.1, 3.1, 6.1, 1.7]; % Imaginary Samples

% Labels 1: Setosa 2: Versicolor 3: Virginica

addpath('../dataset');
dataset = csvread('data.csv');
labels = ["setosa", "versicolor", "virginica"];
% Clean the dataset (Understanding purposes)

sepal_length = dataset(:, 1);
petal_length = dataset(:, 2);
sepal_width  = dataset(:, 3);
petal_width  = dataset(:, 4);
classId      = dataset(:, 5);
X            = [sepal_length, petal_length, sepal_width, petal_width]; % Features
y            = classId;                                                % Variable we want to predict

% Extract features from each class

X_setosa =     X(y == 1, :);
X_versicolor = X(y == 2, :);
X_virginica =  X(y == 3, :);

% Calculate cov. matrices

setosa_covMat = cov(X_setosa);
versicolor_covMat = cov(X_versicolor);
virginica_covMat = cov(X_virginica);

% Extract the means

setosa_mean = mean(X_setosa, 1);
versicolor_mean = mean(X_versicolor, 1);
virginica_mean = mean(X_virginica, 1);

% Calculate the pdf of the samples

setosa_pdf = mvnpdf(samples, setosa_mean, setosa_covMat);
versicolor_pdf =  mvnpdf(samples, versicolor_mean, versicolor_covMat);
virginica_pdf =  mvnpdf(samples, virginica_mean, virginica_covMat);

% Probability of the samples being setosa
prob_setosa = (setosa_pdf)./(setosa_pdf + versicolor_pdf + virginica_pdf);

% Probability of the samples being setosa
prob_versicolor = (versicolor_pdf)./ ...
    (setosa_pdf + versicolor_pdf + virginica_pdf);

% Probability of the samples being setosa
prob_virginica = (virginica_pdf)./ ... 
    (setosa_pdf + versicolor_pdf + virginica_pdf);

matProbabilities = [prob_setosa, prob_versicolor, prob_virginica];

for i=1:size(matProbabilities, 1)
    [value, idx] = max(matProbabilities(i, :));
    disp(['The sample ', num2str(i), ' is a ', char(labels(idx)), ...
        '. Confidence: ', num2str(value*100), ' %']);
end