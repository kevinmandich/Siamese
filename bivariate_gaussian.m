clear all; close all;
scrsz = get(0,'ScreenSize');
figure('Position',[250 scrsz(4)/2-260 scrsz(3)/2 scrsz(4)/1.5])


n_red = 16;  % number of red training points
n_blue = 8;  % number of blue training points
kPrimary = 6;    % number of nearest neighbors to test point
kSub     = 4;    % number of nearest neighbor to current nearest neighbor of test point
plots = 1; % 1 to plot the neighboring radii
inclPrimary = 0; % to include nearest neighbors to original point when calculating sub-neighbors

mu1 = [1 -1]; 
Sigma1 = [.9 .4; .4 .3];
r1 = mvnrnd(mu1, Sigma1, n_blue);

plot(r1(:,1),r1(:,2),'.b');
hold on;

mu2 = [1.5 1.5]; 
Sigma2 = [.3 .2; .2 .9];
r2 = mvnrnd(mu2, Sigma2, n_red);

plot(r2(:,1),r2(:,2),'.r');

% create array of training data
Y = [r1; r2]; % 2 dimensions here
for x = 1:1:length(Y)/2
    Y(x,3) = -1;
    Y(x+length(Y)/2,3) = 1;
end

% create test point
point = [1.15,-0.2];
plot(point(1),point(2),'.k','MarkerSize',25);

neighborSum = knn(point, Y, plots, kPrimary)

neighborSum2 = knn_advanced(point, Y, plots, inclPrimary, kPrimary, kSub)


    
    