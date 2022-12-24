import numpy as np
import matplotlib.pyplot as plt

import lib.mnist_loader as mnist_loader
import sys

import pickle

# Import data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Set and run network
#network_type = "basic"
# network_type = "improved"
#network_type = "advanced"

network_type = "keras"

# Run code
if network_type == "basic":

    import lib.network_basic as network
    
    # Initialize basic network
    net = network.Network([784,30,10])
    net.SGD(training_data, 30, 10, 1.0, test_data=test_data)
    
    
elif network_type == "improved":

    import lib.network_improved as network
    
    filename = "model/Model 30.bin"
    
    train = True
    train_new = True
    
    if train_new:
        net = network.Network([784,30,10])
    else:
        try:
            with open(filename,"rb") as f:
                print("Importing model")
                net = pickle.load(f)
        except:
            print("Model non existing, creating model")
            net = network.Network([784,30,10])
        
    if train:
        # Initialize basic network
        n_epochs=30
        save_each=5
        training_data = list(training_data)
        test_data = list(test_data)
        for i in range(0,n_epochs,save_each): 
            net.SGD(training_data,
                    epochs=save_each, 
                    mini_batch_size=10,
                    eta=.025,
                    lmbda=1e-4,
                    test_data=test_data,
                    cost=network.CrossEntropyCost)
            with open(filename,"wb") as f:
                pickle.dump(net,f)
                print("Saving model")
    else:
        try:
            with open(filename,"rb") as f:
                net = pickle.load(f)
        except:
            sys.exit("Non existing model")
            
        fig = plt.figure()
        ax=fig.add_subplot(111)
        plt.ion()
        plt.show()
        for x,y in test_data:
            ax.imshow(x.reshape(28,28),cmap="binary")
            read_number = np.argmax(net.feedforward(x))
            ax.set_title("Digit: %d - Read digit: %d" %(y,read_number))
            input()
            ax.clear()
    
elif network_type == "advanced":
    
    import lib.network_advanced as network
    
elif network_type == "keras":
    
    import lib.network_keras as network
    import keras
    
    def plot_img(x_test,y_test,net_model):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        plt.ion()
        plt.show()
        
        read_number = net_model.predict(x_test)
        read_number = np.argmax(read_number,axis=1)
        
        # Test model
        for i in range(1000):
            x = x_test[i]
            y = np.argmax(y_test[i])
            x = x.reshape((28,28))
            read_y = read_number[i]
            ax.imshow(x,cmap="binary")
            ax.set_title("Digit: %d - Read digit: %d" %(y,read_y))
            if y==read_y:
                continue
            input()
            ax.clear()
    
    filename = "model/keras 30.bin"
    
    train = False
    train_new = False
    train = True
    # train_new = True
    
    net = network.Network()
    if train_new:
        net.build([(784,0),(100,.5),(10,0)])
    else:
        net.load(filename)
    
    # Format the data in the keras format
    [x_train,y_train] = list(zip(*training_data))
    [x_test,y_test] = list(zip(*test_data))
    x_train = np.array(list(x_train)).reshape((len(x_train),28*28))
    y_train = np.array(list(y_train)).reshape((len(x_train),10))
    x_test = np.array(list(x_test)).reshape((len(x_test),28*28))
    y_test = np.array(keras.utils.to_categorical(y_test, 10))
    
    if train:
        
        net.SGD(x_train,y_train,
                learning_rate = .05,
                epochs = 300,
                batch_size = 12500,
                validation_data = (x_test,y_test))
        
        # Save
        net.save(filename)
        
        plot_img(x_test,y_test,net)
        
    else:
        
        plot_img(x_test,y_test,net)
    
else:
    sys.exit("Not valid network")