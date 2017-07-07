function [] = Gradient_Checking( grads,X,Y,RNN )

h = 1e-4;
numgrads = ComputeGradsNum(X,Y,RNN,h);
for f = fieldnames(RNN)'
    disp('Relative Error between gradients for')
    disp(['Field name: ' f{1} ]);
    Relative_Error(grads.(f{1}),numgrads.(f{1})) 
end

end

