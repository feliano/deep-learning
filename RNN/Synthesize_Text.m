function [ output_sequence ] = Synthesize_Text(RNN,h0,x0,n)
% Genereates an output sequence of one-hot encoded chars with length n
% h0 and x0 should be one-hot encoded

    K = size(RNN.c,1); % no. of unique chars
    output_sequence = zeros(K,n);    
    x = x0;
    
    h=h0;
    for i = 1:n
        a0 = RNN.W*h+RNN.U*x+RNN.b;
        h = tanh(a0);
        o = RNN.V*h + RNN.c;
        p = SoftMax(o);
        
        % randomly sample label from p distribution
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a > 0);
        ii = ixs(1);
        
        % add index to output
        output_sequence(ii,i) = 1;
    
        % draw new value for x
        x = zeros(K,1);
        x(ii,1) = 1;
    end
    
    
    
end
