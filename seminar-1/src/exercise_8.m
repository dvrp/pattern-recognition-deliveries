clearvars;
% Load the data
dataset = [0, 0, 1, -1; 1, -1, 0, 0]

% Convert to polar coordinates
disp('We go from cartesian to polar...');
[theta radius] = cart2pol(dataset(1, :), dataset(2, :));
dataset = [theta; radius]

mu = mean(dataset, 2);
range = max(dataset, [], 2) - min(dataset, [], 2);
range(range == 0) = 1;
% Normalize
Z = (dataset - mu)./range;
covariance_mat = (Z * Z')


% Plotting
figure(1);
polarplot(dataset(1, :), dataset(2, :),'bo','MarkerSize', 10, 'lineWidth', 2);
hold on;
rlim([0, 1.25]);
title('PCA with polar coordinates')


% Calculate covariance matrix
%covariance_mat = cov(dataset')

% Calculate eigenvectors
[eigenVector eigenValue] = eig(covariance_mat)

% Take the most important eigenvectors
disp('We take the vector (in our new polar space) with the highest eigenValue.');
[value, idx] = max(eigenValue);
[value, idx] = max(value);
disp(['Highest Eigenvalue: ', num2str(value)]);
eigenVector_reduced = eigenVector(:, :);
%disp(['Max. var. Eigenvector / new basis: [' num2str(eigenVector_reduced'), ']']);

% Calculate the new values in the new basis
dimReduction = eigenVector_reduced' * dataset

% Reproject the data
projected_data = (eigenVector_reduced * dimReduction)
polarplot(projected_data(1, :), projected_data(2, :),'rx','MarkerSize', 14, 'lineWidth', 2);
legend('2D data', '1D data');