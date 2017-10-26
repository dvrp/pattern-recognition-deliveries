clearvars;
dataset = [0,0; 1,1; 2,3;3,2; 4,4]';
disp('1. Draw the data');
plot(dataset(1, :), dataset(2, :),'bo','MarkerSize', 10, 'lineWidth', 2);

disp('2. Compute the covariance matrix');
mu = mean(dataset, 2);
range = max(max(dataset, [], 2)) - min(min(dataset, [], 2));
% Normalize
Z = (dataset - mu)/range;
covarMat = (Z * Z')

% apply PCA and find the basis where the data has the maximum variance

% Find the eigenvalues and vectors to do so
disp('3. Apply PCA and find the basis:');
[eigenVector, eigenValue] = eig(covarMat)

disp('We take the vector with the highest eigenValue.');
[value, idx] = max(eigenValue);
[value, idx] = max(value);

disp(['Highest Eigenvalue: ', num2str(value)]);
eigenVector_reduced = eigenVector(:, idx);
disp(['Max. var. Eigenvector / new basis: ['...
    num2str(eigenVector_reduced'), ']']);

dimReduction = eigenVector_reduced' * dataset

projected_X = eigenVector_reduced * dimReduction
hold on;
plot(projected_X(1, :), projected_X(2, :), 'r+', 'MarkerSize', ...
    14, 'lineWidth', 2);
hor_axis_helper = min(dataset(1, :)):0.1:max(dataset(2, :));
quiver(0, 0, eigenVector_reduced(1), eigenVector_reduced(2), ...
    1, 'maxHeadSize', 1);
% Add text
txt1 = ['\leftarrow eigenVector: [', num2str(eigenVector_reduced(1)), ...
    ', ', num2str(eigenVector_reduced(1)), ']'];
text(eigenVector_reduced(1), eigenVector_reduced(2),txt1)


plot(hor_axis_helper, hor_axis_helper*eigenVector_reduced(2)/...
    eigenVector_reduced(1));

title('Projection of datapoints using PCA');
legend('Datapoints', 'Projected Datapoints', 'New Basis', ...
    'Projection Axis', 'location', 'northwest');