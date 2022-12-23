import numpy as np
import random

class Network(object):
	
	"""
	Sigmoid neuron
	Quadratic cost function
	"""
	
	def __init__(self,sizes):
		
		self.sizes = sizes
		self.n_layers = len(sizes)
		self.weights = [np.random.normal(0,1,size=(rows,cols))
							for rows,cols in zip(sizes[1:],sizes[:-1])]
		self.biases = [np.zeros((rows,1)) for rows in sizes[1:]]
		
	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		
		# Set parameters for training
		training_data = list(training_data)
		n = len(training_data)
		
		# Set data if there are test data
		if test_data:
			test_data = list(test_data)
			n_test = len(test_data)
		
		# Train the network
		for i in range(epochs):
			# Shuffle data for training
			random.shuffle(training_data)
			# Divide the trainind data in batches of size mini_batch_size
			mini_batches = [training_data[k:k+mini_batch_size]
								for k in range(0,n,mini_batch_size)]
			# Train for one batch
			for mini_batch in mini_batches:
				current_batch_size = len(mini_batch)
				# Compute the variation of the weights for one batch
				grad_w,grad_b = self.mini_batch_update(mini_batch)
				# Update the weights and biases
				self.weights = [w-dw*eta/current_batch_size
									for w,dw in zip(self.weights,grad_w)]
				self.biases = [b-db*eta/current_batch_size
									for b,db in zip(self.biases,grad_b)]
									
									
			# Test data after finishing one epoch
			n_correct = self.evaluate(test_data)
			if test_data:
				print("Epoch {0}: {1}/{2}".format(i,n_correct,n_test))
			else:
				print("Epoch {0}".format(i))
	
	def mini_batch_update(self,batch):
		
		# List of parameters with the gradient for the whole batch
		grad_w = [np.zeros(w.shape) for w in self.weights]
		grad_b = [np.zeros(b.shape) for b in self.biases]
		
		# Propagate one input and study the result
		for x,y in batch:
			delta_w,delta_b = self.backpropagation(x,y)
			grad_w = [gw+dw for gw,dw in zip(grad_w,delta_w)]
			grad_b = [gb+db for gb,db in zip(grad_b,delta_b)]
		
		return grad_w, grad_b
			
	def evaluate(self,data):
		
		results = [(np.argmax(self.feedforward(x)),y)
						for x,y in data]
		return np.sum(int(x==y) for x,y in results)
	
	def feedforward(self,a):
		for w,b in zip(self.weights,self.biases):
			a = sigmoid(np.dot(w,a)+b)
		return a
		
	def backpropagation(self,x,y):
		
		delta_w = [np.zeros(w.shape) for w in self.weights]
		delta_b = [np.zeros(b.shape) for b in self.biases]
		
		activation = x
		activations = [x]
		zs = []
		# Feedforward
		for w,b in zip(self.weights,self.biases):
			z = np.dot(w,activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		
		# Backpropagation
		delta = (activations[-1]-y)*sigmoid_prime(zs[-1])
		delta_b[-1] = delta
		delta_w[-1] = np.dot(delta,activations[-2].transpose())
		
		for i in range(2,self.n_layers):
			delta = np.dot(self.weights[-i+1].transpose(),delta) * \
								sigmoid_prime(zs[-i])
		
			delta_b[-i] = delta
			delta_w[-i] = np.dot(delta,activations[-i-1].transpose())
				
		return delta_w,delta_b

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
