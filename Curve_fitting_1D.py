"""
@author: Yongji Wang
@modified by: Junxiao Zhao
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


class Neural_Network:
    # Initialize the class
    def __init__(self, t_u, x_u, layers, lt, ut):   #自变量、因变量、层数、最小边界、最大边界

        self.t_u = t_u
        self.x_u = x_u

        self.lt = lt
        self.ut = ut

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # Create a list including all training variables
        self.train_variables = self.weights + self.biases
        # Key point: anything updates in train_variables will be 
        #            automatically updated in the original tf.Variable

        # define the loss function
        self.loss = self.loss_NN()

        self.optimizer_Adam = tf.optimizers.Adam()

    '''
    Functions used to establish the initial neural network
    ===============================================================
    '''

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lt) / (self.ut - self.lt) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    '''
    Functions used to building the physics-informed contrainst and loss
    ===============================================================
    '''
    def net_u(self, t):
        x = self.neural_net(t, self.weights, self.biases)
        return x

    @tf.function
    # calculate the physics-informed loss function
    def loss_NN(self):
        self.x_pred = self.net_u(self.t_u)
        loss = tf.reduce_mean(tf.square(self.x_u - self.x_pred))
        return loss

    '''
    Functions used to define ADAM optimizers
    ===============================================================
    '''
    # define the function to apply the ADAM optimizer
    def Adam_optimizer(self, nIter):
        varlist = self.train_variables
        start_time = time.time()
        for it in range(nIter):
            self.optimizer_Adam.minimize(self.loss_NN, varlist)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.loss_NN()
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

    '''
    Function used for training the model
    ===============================================================
    '''
    def train(self, nIter):
        self.Adam_optimizer(nIter)


    def predict(self, t):
        x_p = self.net_u(t)
        return x_p


if __name__ == "__main__":
    noise = 0.0

    np.random.seed(123)
    tf.random.set_seed(123)

    N_tr = 200
    N_pd = 200
    layers = [1, 10, 1]


    def fun_test(t):
        # customize the function by the user
        x = 1 - t ** 2 + tf.exp(-t)
        return x

    t = np.linspace(-1, 1, 200)[:, None]
    x = fun_test(t)
    t_train = tf.cast(t, dtype=tf.float32)
    x_train = tf.cast(x, dtype=tf.float32)

    # Doman bounds
    lt = t.min(0)
    ut = t.max(0)

    model = Neural_Network(t_train, x_train, layers, lt, ut)

    start_time = time.time()
    model.train(5000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    t_intp = np.linspace(-1, 1, N_pd)
    t_intp = tf.cast(t_intp, dtype=tf.float32)
    t_intp = tf.reshape(t_intp, [N_pd, 1])
    x_intp = fun_test(t_intp)

    x_pred = model.predict(t_intp)

    error_x = np.linalg.norm(x_intp - x_pred, 2) / np.linalg.norm(x_intp, 2)
    print('Error u: %e' % error_x)

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    fig = plt.figure(figsize=[10, 10], dpi=200)

    ax = plt.subplot(211)
    ax.plot(t_intp, x_intp, 'b-', linewidth=2, label='Exact')
    ax.plot(t_intp, x_pred, 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$t$', fontsize=15)
    ax.set_ylabel('$x$', fontsize=15, rotation=0)
    ax.set_title('$Fitting$', fontsize=15)

    ax = plt.subplot(212)
    ax.plot(t_intp, x_intp - x_pred, 'b-', linewidth=2)
    ax.set_xlabel('$t$', fontsize=15)
    ax.set_ylabel('f', fontsize=15, rotation=0)

    plt.show()
