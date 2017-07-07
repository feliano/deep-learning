function [ str ] = Decode_One_Hot( one_hot_sequence,ind_to_char )
% Converts a sequence of one-hot encoded characters to an array of chars

str = '';
for i = 1:size(one_hot_sequence,2)
    [row,~] = find(one_hot_sequence(:,i)==1);
    ch = ind_to_char(row);
    %str = strcat(str,ch);
    str = [str ch];
end


end

