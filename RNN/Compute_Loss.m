function [ loss ] = Compute_Loss(X, Y, RNN, h)
% Computes the Cross-entropy loss of for the RNN
    
    tau = size(X,2);
    [ ~,P,~ ] = Forward_Pass( RNN,X,Y,h );
    loss = 0.0;
    for t = 1:tau
        loss = loss + (-log(Y(:,t)'*P(:,t)));
    end
    

end

