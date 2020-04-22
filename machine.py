from tensorflow.keras.layers import*
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.models import Model, load_model
from numpy import array, reshape

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
        gradients = [[0] * len(data) for _ in range(self.nnet.input_shape[1])]
        W = self.nnet.get_weights()
        W1, B, W2= W[0], W[1], W[2]
        H = array(data)*W1
        for i in range(len(gradients)):
            for j in range(len(H)):
                for k in range(len(H[j])):
                    if H[j][k] > -B[k]:
                        gradients[i][j] += W1[i][k] * W2[k][0]
        return gradients
    
    def save(self, name):
        self.nnet.save(name + ".h5")
