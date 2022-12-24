Simple neural network for digits recognition, in particular I used the MNIST database as training data set.
The library consist in 4 files:
- network_basic.py is the most basic implementation of the network without considering strategies to improve the learning speed, accuracy, etc. There is one hidden layer.
- network_improved.py consist in the same code as the previous file, but there are some techniques applied (cross entropy cost function, different initialization, regularization, etc.). There is one hidden layer.
- network_keras.py consist in network_improved.py implemented with Keras. There is one hidden layer that is dropout.
- network_keras_CNN.py makes use of a convolutional neural network to perform the job. There are two couples of convolutional and pooling layers, one layer with all the connections and dropout of 0.5 and the output layer.
