function [neighborSum] = knn_advanced(point, Y, plots, inclPrimary, k1, k2)

% determine distances to each point
tempY = Y;
for i = 1:1:length(tempY)
    tempY(i,4) = sqrt((tempY(i,1) - point(1))^2 + (tempY(i,2) - point(2))^2);
end
% sort distances -- shortest to longest
[sortedDistance(:,4),index] = sort(tempY(:,4));
for j = 1:1:length(tempY)
    sortedDistance(j,3) = tempY(index(j),3);
    sortedDistance(j,1) = tempY(index(j),1);
    sortedDistance(j,2) = tempY(index(j),2);
end

% plot k1 nearest neighbors
for q = 1:1:k1
    if plots == 1: plot(sortedDistance(q,1),sortedDistance(q,2),'ok','MarkerSize',10); end
end

% plot circle around test point, with diameter = distance to furthest
% nearest neighbor
maxDistance = sortedDistance(k1,4);
for q = 1:1:360
    if plots == 1: plot((point(1) + maxDistance*cos(q*pi/180)), (point(2) + maxDistance*sin(q*pi/180)), '.k', 'MarkerSize', 1); end
end

neighborSum = 0;
for q = 1:1:k1
    point = [sortedDistance(q,1), sortedDistance(q,2)];
    if inclPrimary == 0
       for p = 1:1:length(Y)-k1
          reducedY(p,:) = sortedDistance(k1+p,:);
       end
    else
        reducedY = sortedDistance;
    end
    neighborSum = neighborSum + knn(point, reducedY, plots, k2);
end
neighborSum = neighborSum/k1;

end