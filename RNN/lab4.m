% RNN Script
clear all;

% load book data
book_fname = 'data/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);
book_chars = unique(book_data);
% facilitate conversion from/to one-hot
char_to_ind = containers.Map('KeyType','char','ValueType','int32'); 
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

% fill in the indices
for i = 1:length(book_chars)
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end

% settings
m = 100;
eta = 0.1;
seq_length = 25;
sig = 0.01;
K = length(book_chars);
h = zeros(m,1);

% parameters
RNN = struct();
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m, K) * sig;
RNN.W = randn(m, m) * sig;
RNN.V = randn(K, m) * sig;

Adagrad = struct();
Adagrad.b = zeros(m,1);
Adagrad.c = zeros(K,1);
Adagrad.U = zeros(m, K);
Adagrad.W = zeros(m, m);
Adagrad.V = zeros(K, m);

% Used for initial loss computation and gradient checking
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);
[X,Y] = Make_One_Hot( X_chars,Y_chars,char_to_ind );
 
% h0 = zeros(m,1);
% [xentropy,P,H] = Forward_Pass(RNN,X,Y,h0);
% [grads] = Backward_Pass(RNN,X,Y,P,H);
% 
% % Compare analytical and numerical gradients
% Gradient_Checking(grads,X,Y,RNN)

% Train RNN
hprev = zeros(m,1);
nepochs = 2;
eps = 1e-8; % small value to avoid division by zero
smooth_loss = Compute_Loss(X,Y,RNN,hprev);
saved_loss = [];%[smooth_loss];
for e = 1:nepochs
    fprintf("Epoch %d\n",e);
    hprev = zeros(m,1);
    for i = 1:length(book_data)-seq_length-1
        
        % Get the next sequence of chars from the book
        X_chars = book_data(i:i+seq_length-1);
        Y_labels = book_data(i+1:i+seq_length);
        [X,Y] = Make_One_Hot(X_chars,Y_labels,char_to_ind);
        
        % Synthesize Text
        if(mod(i,10000) == 0 || i == 1)
            one_hot_text = Synthesize_Text(RNN,hprev,X(:,1),200);
            text = Decode_One_Hot(one_hot_text,ind_to_char);
            fprintf("%s\n",text);
        end
        
        % Compute Gradients
        [loss,P,H] = Forward_Pass(RNN,X,Y,hprev);
        [grads] = Backward_Pass(RNN,X,Y,P,H);
                
        % adagrad update
        for f = fieldnames(RNN)'
            Adagrad.(f{1}) = Adagrad.(f{1}) + grads.(f{1}).^2;
            RNN.(f{1}) = RNN.(f{1}) - (eta*grads.(f{1}))./sqrt(Adagrad.(f{1})+eps);
        end
        
        smooth_loss = smooth_loss * 0.999 + 0.001 * loss;
        if(mod(i,100) == 0)
            fprintf("iter %d\n",i);
            fprintf("Smooth Loss %f\n",smooth_loss);
            saved_loss = [saved_loss smooth_loss];
        end 
        
        %update hprev
        hprev = H(:,end);
        
    end
end
   
% Plot Loss
figure,
x_axis = linspace(0,1000*(length(saved_loss)-1),length(saved_loss))
plot(x_axis,saved_loss);
title('Smooth Loss over time')
xlabel('Iterations')
ylabel('Loss')
