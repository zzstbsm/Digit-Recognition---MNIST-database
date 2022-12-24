import numpy as np
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import initializers,regularizers,losses,optimizers,metrics

"""
Neural network with keras and ReLU activation function
"""

class Network():
    
    def __init__(self):
        
        return
        
    def build(self,input_shape):
        
        # Create model
        model = Sequential()
        
        # Add input layer
        model.add(Conv2D(32,kernel_size=(5,5),
                            activation="relu",
                            input_shape=input_shape
                            ))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(20,kernel_size=(5,5),
                            activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(100,activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10,activation="softmax"))
        
        self.model = model
    
    def save(self,filename):
        
        self.model.save(filename)
    
    def load(self,filename):
        
        self.model = keras.models.load_model(filename)
        
    def SGD(self,x_train,y_train,learning_rate,epochs,batch_size,validation_data):
        
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