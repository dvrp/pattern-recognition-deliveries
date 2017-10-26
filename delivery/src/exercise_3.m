% Exercise 3. Gaussian model.
% Generates the probability density functions of two assumed variables
% using 1-D Gaussian Equation.

size_var = linspace(0,120, 200);
mu_cat = 40;
mu_puma = 70;
sigma_cat = 12;
sigma_puma = 7;
p_cat = normpdf(size_var, mu_cat, sigma_cat);
p_puma = normpdf(size_var, mu_puma, sigma_puma);
plot(size_var, p_cat, 'lineWidth', 2);
hold on;
plot(size_var, p_puma, 'lineWidth', 2);
title('PDFs of two distributons');
ylabel('probability');
xlabel('Size [cm]');

p_cat_func = @(x) exp(-(x-mu_cat).^2 / (2*sigma_cat^2)) / sqrt(2*sigma_cat^2*pi);
p_puma_func = @(x) exp(-(x-mu_puma).^2 / (2*sigma_puma^2)) / sqrt(2*sigma_puma^2*pi);
intersection = fzero(@(x) p_cat_func(x) - p_puma_func(x), (mu_cat + mu_puma)/2);
disp(['The decision error is around = ', num2str(intersection)]);
line([intersection, intersection], [0, p_cat_func(intersection)], 'Color','red','LineStyle','--', 'lineWidth', 1.5);
plot(intersection, p_cat_func(intersection), 'ro', 'lineWidth', 2);
legend('Cat Prob. Dist.', 'Puma Prob. Dist.', 'Decision boundary', 'Intersection of distributions');

hold off;

% Part 2

disp('Assuming the variance is the same for both distributions...')
figure(2);
size_var = linspace(0,120, 200);
mu_cat = 40;
mu_puma = 70;
sigma_cat = 12;
sigma_puma = 12;
p_cat = normpdf(size_var, mu_cat, sigma_cat);
p_puma = normpdf(size_var, mu_puma, sigma_puma);
plot(size_var, p_cat, 'lineWidth', 2);
hold on;
plot(size_var, p_puma, 'lineWidth', 2);
title('PDFs of two distributons');
ylabel('probability');
xlabel('Size [cm]');

p_cat_func = @(x) exp(-(x-mu_cat).^2 / (2*sigma_cat^2)) / sqrt(2*sigma_cat^2*pi);
p_puma_func = @(x) exp(-(x-mu_puma).^2 / (2*sigma_puma^2)) / sqrt(2*sigma_puma^2*pi);
intersection = fzero(@(x) p_cat_func(x) - p_puma_func(x), (mu_cat + mu_puma)/2);
disp(['The decision error is 0.5 at size = ', num2str(intersection)]);
line([intersection, intersection], [0, p_cat_func(intersection)], 'Color','red','LineStyle','--', 'lineWidth', 1.5);
plot(intersection, p_cat_func(intersection), 'ro', 'lineWidth', 2);
legend('Cat Prob. Dist.', 'Puma Prob. Dist.', 'Decision boundary', 'Intersection of distributions');