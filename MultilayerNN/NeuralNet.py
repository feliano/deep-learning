import os
import numpy as np
import pickle
import sklearn.preprocessing
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import copy
import math


class Cifar10:
	training_file_path = "cifar-10-batches-py/data_batch_";
	test_file_path = "cifar-10-batches-py/test_batch";

	def unpickle(self,file):
	    with open(file, 'rb') as fo:
	        d = pickle.load(fo, encoding='bytes');
	    return d

	# num = 0 means load all data, while entry 1-5 loads only one of the batches
	def load_training_batch(self,num = 0):
		if num == 0:
			#load all data
			img_data = np.ndarray([0,3072]);
			img_labels = np.ndarray([0]);
			for i in range(1,6):
				d = self.unpickle(self.training_file_path + str(i));
				img_data = np.concatenate((img_data,d[b'data']),axis=0);
				img_labels = np.concatenate((img_labels,d[b'labels']),axis=0).astype(int);

			img_data = img_data/255.0;

			return img_data,img_labels;
		elif num > 0 and num < 6:
			d = self.unpickle(self.training_file_path + str(num));
			img_data = d[b'data'];
			img_labels = d[b'labels'];
			img_data = img_data/255.0;
			return img_data,img_labels;
		else:
			raise ValueError("input to load_training_batch() must be between 0-5");

	def load_test_batch(self):
		d = self.unpickle(self.test_file_path);
		img_data = d[b'data'];
		img_labels = d[b'labels'];
		img_data = img_data/255.0;
		return img_data,img_labels;

	# Transforms an array of ground truth labels into a one-hot representation
	def make_one_hot(self,labels):
		binarizer = sklearn.preprocessing.LabelBinarizer();
		binarizer.fit(range(max(labels)+1));
		one_hot_labels = binarizer.transform(labels); 
		return one_hot_labels;

np.random.seed(987234)

class NeuralNetwork:

	def __init__(self):
		print("Neural Network");

		# Load CIFAR data
		print("Loading CIFAR-10 data")
		self.training_data,self.training_labels = Cifar10().load_training_batch(1); # batch 1 is training data
		self.validation_data,self.validation_labels = Cifar10().load_training_batch(2); # batch 2 is used for validation
		self.test_data,self.test_labels = Cifar10().load_test_batch();
		self.training_data = np.matrix(self.training_data.astype('float64'));
		self.validation_data = np.matrix(self.validation_data.astype('float64'));
		self.test_data = np.matrix(self.test_data.astype('float64'));

		print("Training Data:", np.size(self.training_data,0),"x",np.size(self.training_data,1));
		print("Validation Data:", np.size(self.validation_data,0),"x",np.size(self.validation_data,1));
		print("Test Data:", np.size(self.test_data,0),"x",np.size(self.test_data,1));
		print("Training Labels:",len(self.training_labels));
		print("Validation Labels:",len(self.validation_labels));
		print("Test Labels:",len(self.test_labels));

		self.training_one_hot_labels = Cifar10().make_one_hot(self.training_labels);
		self.validation_one_hot_labels = Cifar10().make_one_hot(self.validation_labels);

		# zero mean
		mean_training_data = np.mean(self.training_data,axis=0)
		self.training_data -= mean_training_data;
		self.validation_data -= mean_training_data;
		self.test_data -= mean_training_data;

		# Define Network architecture and init parameters
		self.eta = 0.001 # 0.001
		self.decay_rate = 0.95
		self.lambada = 0.0; # regularization
		self.input_nodes = np.size(self.training_data,1);
		self.output_nodes = 10;
		self.hidden_layers = 1; # this and variable below must be changed to change num_layers correctly
		self.hidden_nodes = [50]; #[50,30] 50 hidden nodes followed by 30
		self.num_layers = self.hidden_layers+1;		

		W,b = self.init_parameters();
		
		# Grid search
		etas = [0.02402]
		lambadas = [0.0001]

		for eta in etas:
			self.eta = eta
			print("eta",eta)
			for l in lambadas:
				self.lambada = l
				print("lambda",l)
				Wstar,bstar,moving_mean,moving_vars = self.train_network(self.training_data,self.training_one_hot_labels,W,b,epochs=10,use_BN=False,lambada=self.lambada)
				train_acc = self.compute_accuracy(self.training_data,self.training_labels,Wstar,bstar,use_BN=False,moving_mean=moving_mean,moving_vars=moving_vars);
				valid_acc = self.compute_accuracy(self.validation_data,self.validation_labels,Wstar,bstar,use_BN=False,moving_mean=moving_mean,moving_vars=moving_vars);
				final_acc = self.compute_accuracy(self.test_data,self.test_labels,Wstar,bstar,use_BN=False,moving_mean=moving_mean,moving_vars=moving_vars);
				print("Train Accuracy",train_acc)
				print("Validation Accuracy",valid_acc)
				print("Test Accuracy", final_acc)
		
		# Gradient Checking
		'''
		self.training_data = self.training_data[0:2,:]
		self.training_one_hot_labels = self.training_one_hot_labels[0:2,:]
		probs,scores,hidden_scores,scores_hat,score_means,score_vars = self.perform_classification(self.training_data,W,b,use_BN=False);
		grads_b,grads_W = self.compute_gradients(self.training_data,self.training_one_hot_labels,probs,W,b,scores,hidden_scores,scores_hat,lambada=self.lambada,use_BN=False,score_means=score_means,score_vars=score_vars)
		ngrads_b,ngrads_W = self.numerical_approx_gradients(self.training_data,self.training_one_hot_labels,W,b,grads_b,grads_W,1e-5,lambada=self.lambada)
		#J = self.compute_cost(self.training_data,self.training_one_hot_labels,W,b,0.0)
		#print("Cost:",J)
		'''

	def init_parameters(self):
		print("Initializing parameters")
		W = [];
		b = [];
		
		for i in range(0,self.num_layers):
			if(i == 0):
				W.append(np.random.normal(loc=0.0,scale=0.001,size=(self.hidden_nodes[0],self.input_nodes))); #input
				b.append(np.zeros((self.hidden_nodes[0],1)));
			elif(i==self.num_layers-1):
				W.append(np.random.normal(loc=0.0,scale=0.001,size=(self.output_nodes,self.hidden_nodes[-1]))); #output
				b.append(np.zeros((self.output_nodes,1)));
			else:
				W.append(np.random.normal(loc=0.0,scale=0.001,size=(self.hidden_nodes[i],self.hidden_nodes[i-1]))); # weights between hidden layers
				b.append(np.zeros((self.hidden_nodes[i],1)));

		return W,b;

	# Train network using MiniBatch SGD
	def train_network(self,X,Y,W,b,epochs,rho=0.95,momentum=True,use_BN=True,lambada=0.0):

		Xbatches,Ybatches = self.generate_batches(X,Y)

		# init momentum
		v_W = [];
		v_b = [];
		for layer in range(0,self.num_layers):
			v_W.append(np.zeros(W[layer].shape))
			v_b.append(np.zeros(b[layer].shape))

		Wstar = copy.deepcopy(W);
		bstar = copy.deepcopy(b);
		training_costs = [];
		validation_costs = [];

		alpha = 0.99 # how much of old average to save
		moving_mean = []
		moving_vars  = []

		training_costs.append(self.compute_cost(X,Y,Wstar,bstar,lambada=lambada,use_BN=False))
		validation_costs.append(self.compute_cost(self.validation_data,self.validation_one_hot_labels,Wstar,bstar,lambada=lambada,use_BN=False))

		for epoch in range(0,epochs):
			for batch in range(0,len(Xbatches)):
				P,scores,hidden_scores,scores_hat,score_means,score_vars = self.perform_classification(Xbatches[batch],Wstar,bstar,use_BN=use_BN)
				grad_b,grad_W = self.compute_gradients(Xbatches[batch],Ybatches[batch],P,W,b,scores,hidden_scores,scores_hat,lambada=lambada,use_BN=use_BN,score_means=score_means,score_vars=score_vars)

				#Update moving averages
				if(use_BN):
					for k in range(0,self.hidden_layers):
						if(epoch ==0 and batch == 0):
							moving_mean.append(score_means[k]);
							moving_vars.append(score_vars[k]);
						else:
							moving_mean[k] = alpha*moving_mean[k] + (1-alpha)*score_means[k];
							moving_vars[k] = alpha*moving_vars[k] + (1-alpha)*score_vars[k];

				for layer in range(0,self.num_layers):
					# Update Parameters with gradients
					Wstar[layer] = Wstar[layer] - np.multiply(self.eta,grad_W[layer]);
					bstar[layer] = bstar[layer] - np.multiply(self.eta,grad_b[layer]);

			        # Update momentum
					if(momentum):
						v_W[layer] = rho * v_W[layer] + self.eta * grad_W[layer];
						v_b[layer] = rho * v_b[layer] + self.eta * grad_b[layer];

						# Update parameters with momentum
						Wstar[layer] = Wstar[layer] - v_W[layer];
						bstar[layer] = bstar[layer] - v_b[layer];

			training_costs.append(self.compute_cost(X,Y,Wstar,bstar,lambada=lambada,use_BN=use_BN,moving_mean=moving_mean,moving_vars=moving_vars))
			validation_costs.append(self.compute_cost(self.validation_data,self.validation_one_hot_labels,Wstar,bstar,lambada=lambada,use_BN=use_BN,moving_mean=moving_mean,moving_vars=moving_vars))
			self.eta = self.eta * self.decay_rate;

		self.plot_cost(training_costs,validation_costs,lambada=lambada)
		return Wstar,bstar,moving_mean,moving_vars

	def plot_cost(self,training_costs,validation_costs,lambada=0.0):
		fig, ax = plt.subplots()
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
		xvals = range(0,len(training_costs))
		training_curve = plt.plot(xvals,training_costs,label="Training Data")
		validation_curve = plt.plot(xvals,validation_costs,label="Validation Data")
		title = "3-layers w/ BN, eta:%.3f, lambda:%.6f" % (0.02402,0.0001)
		plt.title(title)
		plt.legend(loc=1)
		plt.ylabel("Cost")
		plt.xlabel("Epoch")
		plt.show()


	def generate_batches(self,X,Y,batch_size=100):
		num_samples = np.size(X,0);
		num_batches = int(num_samples/batch_size);
		Xbatches = [];
		Ybatches = [];

		for i in range(0,num_batches):
			idx_start = i*batch_size;
			idx_end = (i+1)*batch_size;
			Xbatch = X[idx_start:idx_end,:];
			Ybatch = Y[idx_start:idx_end,:];
			Xbatches.append(Xbatch)
			Ybatches.append(Ybatch)
		return Xbatches,Ybatches;

	def perform_classification(self,X,W,b,use_BN=True,moving_mean=None,moving_vars=None):
		scores = []
		hidden_scores = []
		score_means = []
		score_vars = []
		scores_hat = []
		num_samples = np.size(X,0)

		for i in range(0,self.hidden_layers):
			if(i == 0):
				score = self.compute_score(X.T,W[i],b[i]);
				scores.append(score);
			else:
				score = self.compute_score(hidden_scores[i-1],W[i],b[i]);
				scores.append(score);

			if(use_BN):
				# batch normalize
				s_mean = np.mean(scores[i],axis=1);
				score_means.append(s_mean);
				s_var = np.var(scores[i],axis=1);
				s_var = s_var * ((num_samples-1) / num_samples); # number of samples in batch
				score_vars.append(s_var);

				if(moving_mean==None and moving_vars==None):
					scores_hat.append(self.batch_normalize(scores[i],score_means[i],score_vars[i]));		
				else:
					moving_mean[i]
					moving_vars[i]
					scores_hat.append(self.batch_normalize(scores[i],moving_mean[i],moving_vars[i])); # use moving averages for test

				hidden_score = self.relu(scores_hat[i]);
				hidden_scores.append(hidden_score);
			else:
				hidden_score = self.relu(score);
				hidden_scores.append(hidden_score);

		# last layer do softmax
		final_score = self.compute_score(hidden_scores[-1],W[-1],b[-1])		
		P = self.softmax(final_score)
		return P,scores,hidden_scores,scores_hat,score_means,score_vars
	
	# score should be shape (output_nodes,num_samples)
	def softmax(self,score):
		val = np.exp(score)/(np.sum(np.exp(score),axis=0))
		return val

	def relu(self,score):
		return np.maximum(0,score);

	def compute_score(self,X,W,b):
		return W*X+b;

	def compute_gradients(self,X,Y,P,W,b,scores,hidden_scores,scores_hat,lambada=0.0,use_BN=True,score_means=None,score_vars=None):
		grads_W = []
		grads_b = []

		g = []
		num_samples = len(Y)

		# preallocate gradient matrices
		for l in range(0,self.num_layers):
			grad_b = np.matrix(np.zeros(b[l].shape))
			grads_b.append(grad_b)
			grad_W = np.matrix(np.zeros(W[l].shape))
			grads_W.append(grad_W)
			g.append(np.matrix(np.zeros((num_samples,np.size(b[l],0)))))

		# last layer
		for i in range(0,num_samples):
			y = Y[i,:]
			p = P[:,i]
			
			g[-1][i,:] = -(y-p.T)
			grads_b[-1] += g[-1][i,:].T
			grads_W[-1] += g[-1][i,:].T * hidden_scores[-1][:,i].T

			s = scores[-1][:,i]
			if(use_BN==True and scores_hat != None):
				s = scores_hat[-1][:,i] # USE BN
			s[s>0] = 1;
			s[s<=0] = 0;
			gtemp = g[-1][i,:]*W[-1]
			g[-2][i,:] = gtemp*np.diagflat(s)

		grads_b[-1] = (1/num_samples) * grads_b[-1]  
		grads_W[-1] = (1/num_samples) * grads_W[-1] + 2.0*lambada*W[-1]

		# rest of the layers
		for l in range(0,self.num_layers-1):
			index = -(l+1) # to get correct nodes from hidden_nodes
			current_layer = self.num_layers-l-2

			if(use_BN==True and score_means!=None and score_vars!=None):
				g[current_layer] = self.batchnorm_backpass(g[current_layer],scores[current_layer],score_means[current_layer],score_vars[current_layer]); # returns dJ/ds

			for i in range(0,num_samples):
				grads_b[current_layer] += g[current_layer][i,:].T
				if(current_layer == 0):
					grads_W[current_layer] += np.matrix(g[0][i,:]).T * np.matrix(X[i,:])
				else:
					grads_W[current_layer] += np.matrix(g[current_layer][i,:]).T * np.matrix(hidden_scores[current_layer-1][:,i]).T 
		
			# propagate g to next layer for networks with num_layers > 2
			if(current_layer > 0):
				s = scores[current_layer-1][:,i]
				if(use_BN==True and scores_hat != None):
					s = scores_hat[current_layer-1][:,i]
				s[s>0] = 1;
				s[s<=0] = 0;
				gtemp = g[current_layer][i,:]*W[current_layer]
				g[current_layer-1][i,:] = gtemp*np.diagflat(s)

			grads_b[current_layer] = (1/num_samples) * grads_b[current_layer];
			grads_W[current_layer] = (1/num_samples) * grads_W[current_layer] + 2.0*lambada*W[index-1]
			
		return grads_b,grads_W

	def cross_entropy(self,y,probs):
		y = np.matrix(y)
		probs = np.matrix(probs)
		return -np.log(y*probs)


	def compute_cost(self,X,Y,W,b,lambada=0.0,use_BN=True,moving_mean=None,moving_vars=None):

		P,_,_,_,_,_ = self.perform_classification(X,W,b,use_BN=use_BN,moving_mean=moving_mean,moving_vars=moving_vars)
		num_samples = len(Y)

		cross_entropy = 0.0
		for i in range(0,num_samples):
			y = Y[i,:]
			p = P[:,i]
			cross_entropy += self.cross_entropy(y,p)

		regularization = 0.0
		for l in range(0,self.num_layers):
			regularization += np.sum(np.multiply(W[l],W[l]))

		J = (1/num_samples) * cross_entropy + lambada * regularization

		return J[0,0]

	# compares predictions with given labels
	def compute_accuracy(self,X,y,W,b,use_BN=True,moving_mean=None,moving_vars=None):
		print("Accuracy")
		num_samples = np.size(X,0)
		if(use_BN):
			P,_,_,_,_,_ = self.perform_classification(X,W,b,use_BN=use_BN,moving_mean=moving_mean,moving_vars=moving_vars);
		else:
			P,_,_,_,_,_ = self.perform_classification(X,W,b,use_BN=use_BN,moving_mean=None,moving_vars=None);

		# get argmax
		indices = np.argmax(P,axis=0)
		correct_labels = 0;
		for i in range(0,np.size(indices,1)):
			if(indices[0,i] == y[i]):
				correct_labels += 1;

		accuracy = correct_labels/num_samples;
		return accuracy

	def batch_normalize(self,scores,score_means,score_vars):
		epsilon = 1e-6; # to avoid division by zero
		mean_subtracted_scores = (scores-score_means);
		root_vars = np.sqrt(score_vars + epsilon)
		scores_hat = mean_subtracted_scores / root_vars
		return scores_hat

	# input g represents partial deriv. of s_hat wrt. cost J.
	# output g represents partial deriv. of s wrt.cost J.
	def batchnorm_backpass(self,g_in,s,mean,var):
		dJVb = 0;
		dJub = 0; 

		g_out = np.matrix(np.zeros(g_in.shape));
		
		for i in range(0,np.size(g_in,0)):
			p_1 = np.multiply(g_in[i,:],np.power(var,(-3/2)).T)
			p_2 = np.diagflat(s[:,i]-mean)
			p_res = (-0.5)*p_1*p_2
			dJVb += (-0.5)*p_1*p_2
			res = (np.multiply(g_in[i,:],np.power(var,(-1/2)).T))
			dJub += (-1.0) * np.multiply(g_in[i,:],np.power(var,(-1/2)).T)

			p1 = np.multiply(g_in[i,:],(np.power(var,(-0.5)).T));
			p2 = ((2.0/np.size(g_in,1))*dJVb*np.diagflat(s[:,i]-mean))
			p3 = dJub * (1.0/np.size(g_in,1));
			gtemp = p1+p2+p3;
			g_out[i,:] = gtemp

		return g_out

	def gradient_checking(self,grads_b,grads_W,ngrads_b,ngrads_W):
		# Compute relative error
		errors_b = []
		errors_W = []
		for i in range(0,np.size(grads_b,0)):
			err_b = np.abs(grads_b[i]-ngrads_b[i])/(np.abs(grads_b[i])+np.abs(ngrads_b[i]))
			err_W = np.abs(grads_W[i]-ngrads_W[i])/(np.abs(grads_W[i])+np.abs(ngrads_W[i]))
			errors_b.append(err_b)
			errors_W.append(err_W)
		return

	def relative_error(self,grad,ngrad):
		return np.abs(ngrad-grad)/(np.abs(grad)+np.abs(ngrad))

	def numerical_approx_gradients(self,X,Y,W,b,grads_b,grads_W,h,lambada=0.0):
		ngrads_b = []
		ngrads_W = []

		# Check bias gradients 
		for j in range(0,self.num_layers):
			ngrad_b = np.matrix(np.zeros(np.size(b[j])))
			for i in range(0,np.size(b[j],0)):
				b_try = copy.deepcopy(b) # copy list and objects contained
				b_try[j][i,0] = b_try[j][i,0] - h
				c1 = self.compute_cost(X,Y,W,b_try,use_BN=False,lambada=lambada)

				b_try = copy.deepcopy(b)
				b_try[j][i,0] = b_try[j][i,0] + h
				c2 = self.compute_cost(X,Y,W,b_try,use_BN=False,lambada=lambada)

				ngrad_b[0,i] = (c2-c1) / (2*h)
				err = self.relative_error(grads_b[j][i,0],ngrad_b[0,i])

				print("Grad_b: %f - NGrad_b: %f Relative Error:%f" %(grads_b[j][i,0],ngrad_b[0,i],err));

			ngrads_b.append(ngrad_b);
		
		# Check weight gradients 		
		for j in range(0,self.num_layers):
			ngrad_W = np.matrix(np.zeros(W[j].shape))
			for i in range(0,np.size(W[j],0)):
				for k in range(0,np.size(W[j],1)):				
					print(i,k)
					W_try = copy.deepcopy(W);
					W_try[j][i,k] = W_try[j][i,k] - h;
					c1 = self.compute_cost(X,Y,W_try,b,lambada=lambada)
					W_try = copy.deepcopy(W);
					W_try[j][i,k] = W_try[j][i,k] + h;
					c2 = self.compute_cost(X,Y,W_try,b,lambada=lambada)
					ngrad_W[i,k] = (c2-c1) / (2*h);
					err = self.relative_error(grads_W[j][i,k],ngrad_W[i,k])
					print("Grad: %f - NGrad: %f Relative Error:%f" %(grads_W[j][i,k],ngrad_W[i,k],err));

			ngrads_W.append(ngrad_W)
		return ngrads_b,ngrads_W

nn = NeuralNetwork()
