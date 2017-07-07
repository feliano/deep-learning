function [  ] = Relative_Error( grad,numgrad )
% Computes relative error between numerical and analytical gradients

rel = abs(grad(:)-numgrad(:))/(abs(grad(:))+abs(numgrad(:)));
max(rel(:))

end

