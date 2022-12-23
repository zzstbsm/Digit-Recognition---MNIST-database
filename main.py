import numpy as np
import lib.mnist_loader as mnist_loader

# Import data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Set and run network
#network_type = "basic"
network_type = "improved"
#network_type = "advanced"

# Run code
if network_type == "basic":

	import lib.network_basic as network
	
	# Initialize basic network
	net = network.Network([784,30,10])
	net.SGD(training_data, 30, 10, 1.0, test_data=test_data)
	
	
elif network_type == "improved":

	import lib.network_improved as network
	
	# Initialize basic network
	net = network.Network([784,30,10])
	net.SGD(training_data,
			epochs=200, 
			mini_batch_size=10,
			eta=.05,
			lmbda=5e-4,
			test_data=test_data,
			cost=network.CrossEntropyCost)

elif network_type == "advanced":
	import lib.network_advanced as network
else:
	print("Not valid network")



