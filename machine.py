from tensorflow.keras.layers import*
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.models import Model, load_model
from numpy import reshape

class Machine():
    
    def __init__(self, ins = None, model = None, nodes = 1024, lr = 0.01):
        if model:
            self.nnet = load_model(model + ".h5")
        elif ins:
            X = Input(ins)
            
            H = Activation('relu')(Dense(nodes)(X))
            
            Y = Dense(1)(H)
            
            self.nnet = Model(inputs = X, outputs = Y)
            self.nnet.compile(
                optimizer = Adam(learning_rate = lr),
                loss = 'mean_squared_error'
            )
        else:
            raise ValueError("Provide either input shape or model name")
    
    def learn(self, X, Y, ep = 32):
        self.nnet.fit(X, Y, epochs = ep)
    
    def v(self, data):
        return self.nnet.predict(data)
    
    def d(self, data):
        slopes = [0] * len(data)
        W = self.nnet.get_weights()
        W1, B, W2 = W[0], W[1], W[2]
        W1 = W1[0]
        W2 = reshape(W2, (len(W2),))
        W = W1*W2
        for i in range(len(slopes)):
            for j in range(len(W1)):
                if W1[j] * data[i] > -B[j]:
                    slopes[i] += W[j]
        return slopes
    
    def save(self, name):
        self.nnet.save(name + ".h5")
