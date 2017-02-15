function [closestMean, meanLocs, nIters ] = myKMeans( data, k, maxIters )
%KMEANS Runs kmeans algorithm.


dist = @(x,y)(sqrt(sum((x-y).^2))); 
   %%% argmin function
   %%% argmin([2,1,3]) == 2
    function x = argmin(y)
        [~,p] = min(y);
        x = p;
    end
    %% custom compare function that takes advange
    %% of the fact that empty cells exist which also means no change
    function b = cellcmp( x, y)
        if isempty(y)
            b = true;
        else
            b = all(x == y);
        end
    end
    %%% This function ensure that the builtin mean will not 
    %%% collapse since data point into one number
    function o = columnMean( x)
        if size(x,1) > 1
            o = mean(x);
        else
            o = x;
        end
    end
% find how many data point there are
num_points = size(data,1);
% Initialize means randomly among points

%% find the min and max of the data set
%max_data = max(data);
%min_data = min(data);

%% randomly choose a number between min and max found above for each k
%k_means = arrayfun(@(x)(min_data+rand(1,length(max_data)).*(max_data-min_data)), 1:k, 'UniformOutput', false);

%% choose random unique points for kmeans. This map will produce a cell
%% array of k means
k_means = arrayfun(@(x)(data(x,:)),randperm(num_points,k), 'UniformOutput', false);


% Iterate through a max number of times
for i=1:maxIters
    %% set a iter for output maxIter
    iter = i;
    % Find the distance to each mean
    
    %% create anymous function that find the distance to each mean
    all_dist = @(x)(arrayfun(@(z)( dist(k_means{z},x )),1:k));
    %% map the anymous function to all data points
    distances = arrayfun(@(x)( all_dist(data(x,:)) ), 1:num_points, 'UniformOutput', false );
    % Label it to be with the closest mean
    labels = arrayfun(@(x)( argmin(distances{x})), 1:num_points);
    
    %% this line combines Find the diance to each mean and Label
    %% it to be with the closest mean
    %labels = arrayfun(@(x)( argmin(all_dist(data(x,:)))), 1:num_points);
    % Create new means
    % initalize k as an empty cell
    k_means_new = cell(1,k);
    %% loop through all labels to set a new cell array with all the change
    %% k centers
    for m = unique(labels)
        k_means_new{m} = columnMean( data(find(labels==m),:));
    end
    % Check if means do not move!
    not_changed = all( cellfun(@cellcmp, k_means, k_means_new ));
    
    % update the new means
    for m = unique(labels)
        k_means{m} = k_means_new{m};
    end
    % Quit if means do not move! Woooohoooo!
    
    if not_changed == true
        break;
    end
    
    % If they do not move, do it all again!
end
%% change cell array to a proper k matrix
kmat =cell2mat(k_means');
% Plot output as a bunch of points with different colors for each cluster.
%%gscatter(data(:,1),data(:,2), labels); the prof wanted a different output
%% use his function

plotMyDots(data,kmat, labels)
% Prepare output.
closestMean = labels;
meanLocs = kmat;
nIters = iter;
end

