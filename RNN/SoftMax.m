function [ p ] = SoftMax( score )
% Softmax, ensures that each column of p will sum to 1
    p = exp(score)/sum(exp(score),1);
end

