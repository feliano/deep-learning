function [ grads ] = Backward_Pass( RNN,X,Y,P,H )
%Backward_Pass 
%   Backpropagation of gradients

tau = size(X,2);

% derivatives of loss w.r.t. to W,U,V,b,c respectively
dW = zeros(size(RNN.W));
dU = zeros(size(RNN.U));
dV = zeros(size(RNN.V));
db = zeros(size(RNN.b));
dc = zeros(size(RNN.c));
do = zeros(size(RNN.c));

for t = 1:tau
    do(:,t) = -(Y(:,t)-P(:,t));
    dV = dV + do(:,t)*H(:,t)';
    dc = dc + do(:,t);
end

dh = do(:,end)'*RNN.V;
da = zeros(size(RNN.W,1),1)';
h0 = zeros(size(H,1),1);
for t = tau:-1:1
    dh = do(:,t)'*RNN.V + da*RNN.W;
    da = dh*diag(1-H(:,t).^2);
    ind = t-1;
    if(ind == 0)
        dW = dW + da'*h0';
    else        
        dW = dW + da'*H(:,ind)';
    end
        
    dU = dU + da'*X(:,t)';
    db = db + da';
end


grads = struct();
grads.W = dW;
grads.U = dU;
grads.V = dV;
grads.b = db;
grads.c = dc;

% clip gradients to avoid 'exploding gradient problem'
for f = fieldnames(RNN)'
    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
end

end