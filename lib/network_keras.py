import numpy as np
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import initializers,regularizers,losses,optimizers,metrics

"""
Neural network with keras and ReLU activation function
"""

class Network():
    
    def __init__(self):
        
        return
        
    def build(self,sizes):
        
        # Create model
        model = Sequential()
        
        chosen_init = initializers.RandomNormal(
                            mean=0.,
                            stddev = 1./np.sqrt(sizes[0][0]),
                            seed=None
                            )
        chosen_regularizer = regularizers.l2(1e-4)
        
        # Add input layer
        model.add(Dense(
            sizes[1][0],
            activation = "relu", 
            input_shape = (sizes[0][0],),
            kernel_initializer=chosen_init,
            kernel_regularizer=chosen_regularizer,
            name="Input"
            )) 
        model.add(Dropout(sizes[1][1]))
        
        # Hidden layers
        for i in range(2,len(sizes)-1):
            chosen_init = initializers.RandomNormal(
                            mean=0.,
                            stddev = 1./np.sqrt(sizes[0][0]),
                            seed=None
                            )
            model.add(Dense(
                            sizes[i+1][0], 
                            activation = "relu",
                            kernel_initializer=chosen_init,
                            kernel_regularizer=chosen_regularizer,
                            name="Hidden layer"
                            ))
            model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(sizes[-1][0],activation = "softmax",name="Output layer"))
        model.summary()
        self.model = model
    
    def save(self,filename):
        
        self.model.save(filename)
    
    def load(self,filename):
        
        self.model = keras.models.load_model(filename)
        
    def SGD(self,x_train,y_train,learning_rate,epochs,batch_size,validation_data):
        
        # keras.optimizers.SGD(
            # learning_rate = learning_rate,
            # momentum = momentum,
            # nesterov = False
            # )
            
        self.model.compile(
            optimizer = optimizers.SGD(
                            lr = learning_rate
                            ),
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
            )
        
        self.history = self.model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,
                                        validation_data=validation_data)
                                        
    def predict(self,x):
        return self.model.predict(x)