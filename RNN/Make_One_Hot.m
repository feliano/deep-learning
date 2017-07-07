function [ X,Y ] = Make_One_Hot( X_chars,Y_chars,char_to_ind )
% Converts arrays of chars into one-hot encoded sequences
seq_length = size(X_chars,2);
K = size(char_to_ind,1);

X = zeros(K,seq_length);
Y = zeros(K,seq_length);
for i = 1:seq_length
    indx = char_to_ind(X_chars(i));
    X(indx,i) = 1;
    indy = char_to_ind(Y_chars(i));
    Y(indy,i) = 1;
end

end

