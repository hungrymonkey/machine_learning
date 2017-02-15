function [linearDecisionBoundary, didConverge ] = myPerceptron( data, maxIters )


%% Initialize variables
% set n and p according to the description above.
[n, pPlusOne] = size(data);
p = pPlusOne-1;
% Initialize output variables
didConverge = 0;
% Initialize weights and bias terms.
linearDecisionBoundary = ones(1,p+1);


for l=1:maxIters
    for i=1:n
        % This line calculates the predicted output preceptron y
        % Requirements 
        %  linearDecisionBoundary(1:p) = [w1, w2, w3, w4...wp] weights
        %  linearDecisionBoundary(end) = wn bias term
        %  data(i,1:p) = [x1i,x2i,x3i,x4i...] inputs
        %  outputs
        %  1 or -1 preceptron
        y = sign(data(i,1:p) * linearDecisionBoundary(1:p)' + linearDecisionBoundary(end));
        % grab the expected output from the data matrix
        expected = data(i,end);
        % check if the predicted value is correct
        % if true, then update the new weights
        if y ~= expected
            % This line calculates the predicted output preceptron y
            % Requirements 
            % linearDecisionBoundary(1:p) = [w1, w2, w3, w4...wp] weights
            % linearDecisionBoundary(end) = wn bias term
            % output
            % store linearDecisionBoundary = [w1new, w2new, w3new, w4new...wpnew, wn ]
            linearDecisionBoundary(1:p) = linearDecisionBoundary(1:p)...
            + ( (expected-y) * ones(1,p)  ).*data(i,1:p);

            % This line calculates the predicted output preceptron y
            % Requirements 
            % linearDecisionBoundary(1:p) = [w1, w2, w3, w4...wp] weights
            % linearDecisionBoundary(end) = wn bias term
            % output
            % store linearDecisionBoundary(n) = [w1, w2, w3, w4...wp, wnew ]
            linearDecisionBoundary(end) = linearDecisionBoundary(end) + (expected-y);
        end
    end
    % This map assigns a predict label to each data point
    % linearDecisionBoundary(1:p) = [w1, w2, w3, w4...wp] weights
    % linearDecisionBoundary(end) = wn bias term
    % data(1:p) = [x1,x2,x3..xp]
    % output
    % 1xp mat of 1 and -1 predicted values

    % these line does the same thing as below
    %out_tmp = arrayfun(@(x)(sign( data(x,1:p)* linearDecisionBoundary(1:p)' + linearDecisionBoundary(end) )),1:n);

    % this line takes advantage of the fact matrix multiplication diagonal
    % is the dot product row an * column bn
    out_tmp = sign( diag(data(:,1:p)* repmat(linearDecisionBoundary(1:p)',[1,n]))' + linearDecisionBoundary(end));
    %compare two list of labels to see if data converges
    converge = all(out_tmp == data(:,end)');
    % break if it converges
    output = out_tmp;

    

    
    figure(1);
scatter(data(:,1),data(:,2),[],data(:,end),'filled'); % Could use end or p+1, would be the same thing here.
title('target');
figure(2);
scatter(data(:,1),data(:,2),[],output,'filled');
title('model output');
pause(2);
    
    
    if converge == 1
        break
    end
end
%set output variables
didConverge = converge;

output = out_tmp;


%set output variables
%output = linearDecisionBoundary;



end

