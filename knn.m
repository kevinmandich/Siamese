function [neighborSum] = knn(point, Y, plots, k)

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

% plot k nearest neighbors
if sortedDistance(1,4) == 0
    skipFlag = 1;
else
    skipFlag = 0;
end
for q = 1:1:k
    if skipFlag == 1
        if plots == 1: plot(sortedDistance(q+skipFlag,1),sortedDistance(q+skipFlag,2),'om','MarkerSize',12); end
    else
        if plots == 1: plot(sortedDistance(q+skipFlag,1),sortedDistance(q+skipFlag,2),'om','MarkerSize',10); end
    end
end

% plot circle around test point, with diameter = distance to furthest
% nearest neighbor
maxDistance = sortedDistance(k+skipFlag,4);
for q = 1:1:180
    if skipFlag == 1
        if plots == 1: plot((point(1) + maxDistance*cos(q*pi/90)), (point(2) + maxDistance*sin(q*pi/90)), '.m', 'MarkerSize', 1); end
    else
        if plots == 1: plot((point(1) + maxDistance*cos(q*pi/90)), (point(2) + maxDistance*sin(q*pi/90)), '.m', 'MarkerSize', 1); end
    end
end

% determine weight
neighborSum = 0;
for q = 1:1:k
    neighborSum = neighborSum + sortedDistance(q+skipFlag,3);
end
neighborSum = neighborSum/k;

end