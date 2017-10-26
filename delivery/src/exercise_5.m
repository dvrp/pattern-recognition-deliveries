% Exercise 5

scale = 3;
s = scale;
res = 40;

figure(1)

% Chicken?
meanWeight_chicken = 54;
meanHeight_chicken = 5;
mu = [meanWeight_chicken, meanHeight_chicken]; 
SIGMA = [5 .1; .1 .5]; 
[X1,X2] = meshgrid(linspace(meanWeight_chicken - s*sqrt(SIGMA(1, 1)), meanWeight_chicken+s*sqrt(SIGMA(1, 1)),res)',linspace(meanHeight_chicken-s*sqrt(SIGMA(2, 2)),meanHeight_chicken+s*sqrt(SIGMA(2, 2)),res)');
X = [X1(:) X2(:)];
p = mvnpdf(X,mu,SIGMA);
subplot(2, 1, 1);
surf(X1,X2,reshape(p,res,res));
hold on;
X = [60, 5];
p_chicken = mvnpdf(X,mu,SIGMA);
plot3(X(1), X(2), p_chicken, 'r*');
title(['Chicken pdf at point = ' num2str(mvnpdf(X,mu,SIGMA)*100)])
axis tight
view(45, 45)

% Goose?
meanWeight_goose = 65;
meanHeight_goose = 6;
mu = [meanWeight_chicken, meanHeight_chicken]; 
SIGMA = [8 .2; .2 1]; 
[X1,X2] = meshgrid(linspace(meanWeight_chicken - s*sqrt(SIGMA(1, 1)), meanWeight_chicken+s*sqrt(SIGMA(1, 1)),res)',linspace(meanHeight_chicken-s*sqrt(SIGMA(2, 2)),meanHeight_chicken+s*sqrt(SIGMA(2, 2)),res)');
X = [X1(:) X2(:)];
p = mvnpdf(X,mu,SIGMA);
subplot(2, 1, 2);
surf(X1,X2,reshape(p,res,res));
hold on;
X = [60, 5];
p_goose = mvnpdf(X,mu,SIGMA);
plot3(X(1), X(2), p_goose, 'r*', 'lineWidth', 2);
title(['Goose pdf at point = ' num2str(mvnpdf(X,mu,SIGMA)*100)])
axis tight
view(45, 45)
disp(['Probability of being a goose is ' num2str(p_goose/(p_goose+p_chicken))]);

% Top Views

figure(2)


% Chicken?
meanWeight_chicken = 54;
meanHeight_chicken = 5;
mu = [meanWeight_chicken, meanHeight_chicken]; 
SIGMA = [5 .1; .1 .5]; 
[X1,X2] = meshgrid(linspace(meanWeight_chicken - s*sqrt(SIGMA(1, 1)), meanWeight_chicken+s*sqrt(SIGMA(1, 1)),res)',linspace(meanHeight_chicken-s*sqrt(SIGMA(2, 2)),meanHeight_chicken+s*sqrt(SIGMA(2, 2)),res)');
X = [X1(:) X2(:)];
p = mvnpdf(X,mu,SIGMA);
subplot(2, 1, 1);
surf(X1,X2,reshape(p,res,res));
hold on;
X = [60, 5];
p_chicken = mvnpdf(X,mu,SIGMA);
plot3(X(1), X(2), p_chicken, 'r*');
title(['Chicken pdf at point = ' num2str(mvnpdf(X,mu,SIGMA)*100)])
view(0, 90)
axis tight

% Goose?
meanWeight_goose = 65;
meanHeight_goose = 6;
mu = [meanWeight_chicken, meanHeight_chicken]; 
SIGMA = [8 .2; .2 1]; 
[X1,X2] = meshgrid(linspace(meanWeight_chicken - s*sqrt(SIGMA(1, 1)), meanWeight_chicken+s*sqrt(SIGMA(1, 1)),res)',linspace(meanHeight_chicken-s*sqrt(SIGMA(2, 2)),meanHeight_chicken+s*sqrt(SIGMA(2, 2)),res)');
X = [X1(:) X2(:)];
p = mvnpdf(X,mu,SIGMA);
subplot(2, 1, 2);
surf(X1,X2,reshape(p,res,res));
hold on;
X = [60, 5];
p_goose = mvnpdf(X,mu,SIGMA);
plot3(X(1), X(2), p_goose, 'r*');
title(['Goose pdf at point = ' num2str(mvnpdf(X,mu,SIGMA)*100)])
view(0, 90)
axis tight