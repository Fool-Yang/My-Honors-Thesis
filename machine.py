from tensorflow.keras.layers import *
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

    # train
    def learn(self, X, Y, ep = 32, bs = 256):
        self.nnet.fit(X, Y, epochs = ep, batch_size = bs)

    # compute output
    def v(self, data):
        return self.nnet.predict(data)

    # compute gradients at given input data points
    # only works for one hidden layer for now
    def d(self, data):
        gradients = [[0] * len(data) for _ in range(self.nnet.input_shape[1])]
        W = self.nnet.get_weights()
        W1, B, W2= W[0], W[1], W[2]
        H = array(data) @ W1
        for i in range(len(gradients)):
            for j in range(len(H)):
                for k in range(len(H[j])):
                    if H[j][k] > -B[k]:
                        gradients[i][j] += W1[i][k] * W2[k][0]
        return array(gradients)

    # save the model as .h5 file
    def save(self, name):
        self.nnet.save(name + ".h5")
