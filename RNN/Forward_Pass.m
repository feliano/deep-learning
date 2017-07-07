function [ xentropy,P,H ] = Forward_Pass( RNN,X,Y,h0)
%   Forward pass for the RNN
%   X - one-hot encoded sequence of chars
%   Y - one-hot encoded sequence of chars
%   RNN - object holding the network parameters

    K = size(RNN.c,1);
    seq_length = size(X,2);
    P = zeros(K,seq_length);
    H = zeros(size(h0,1),seq_length+1);
    H(:,1) = h0;
    xentropy = 0.0;
    for t = 1:seq_length
        a = RNN.W*H(:,t)+RNN.U*X(:,t)+RNN.b;
        H(:,t+1) = tanh(a);
        o = RNN.V*H(:,t+1)+RNN.c;
        P(:,t) = SoftMax(o);
        
        xentropy = xentropy + (-log(Y(:,t)'*P(:,t)));
    end
    
    H = H(:,2:end);
    
end

