import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data


fashion_mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

#print (fashion_mnist)

label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}
#First hidden layer
n_hidden_1 = 128

#Second hidden layer
n_hidden_2 = 128

n_input = 784
n_classes = 10
n_samples = fashion_mnist.train.num_examples

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name = "X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X,Y

def initalize_parameters():
    # First hidden layer
    W1 = tf.get_variable("W1", [n_hidden_1, n_input], initializer= tf.contrib.layers.xavier_initializer(seed = 42))
    b1 = tf.get_variable("b1", [n_hidden_1, 1], initializer= tf.zeros_initializer())

    #Second hidden layer
    W2 = tf.get_variable("W2", [n_hidden_2, n_hidden_1], initializer= tf.contrib.layers.xavier_initializer(seed = 42))
    b2 = tf.get_variable("b2", [n_hidden_2, 1], initializer= tf.zeros_initializer())

    #Output layer
    W3 = tf.get_variable("W3", [n_classes, n_hidden_2], initializer= tf.contrib.layers.xavier_initializer(seed = 42))
    b3 = tf.get_variable("b3", [n_classes, 1], initializer= tf.zeros_initializer())

    parameters = {
        "W1" : W1,
        "b1": b1,
        "W2": W2,
        "W3": W3,
        "b2": b2,
        "b3": b3
    }

    return parameters

def forward_propagation(X, paramters):
    W1 = paramters['W1']
    b1 = paramters['b1']
    W2 = paramters['W2']
    b2 = paramters['b2']
    W3 = paramters['W3']
    b3 = paramters['b3']

    Z1 = tf.add(tf.matmul(W1,X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost

def model(train, test, learning_rate = 0.0001, num_epochs = 16, minibatch_size = 32, print_cost = True, graph_filename = 'costs'):
    ops.reset_default_graph()
    tf.set_random_seed(42)
    seed = 42

    (n_x, m)= train.images.T.shape
    n_y = train.labels.T.shape[0]

    costs = []

    X,Y = create_placeholders(n_x, n_y)

    parameters = initalize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/ minibatch_size)
            seed = seed + 1

            for i in range(num_minibatches):
                minibatch_X, minibatch_Y = train.next_batch(minibatch_size)

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X.T, Y: minibatch_Y.T})

                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True:
                print("Cost after epoch {epoch_num}: {cost}". format(epoch_num=epoch, cost=epoch_cost))
                costs.append(epoch_cost)


        parameters = sess.run(parameters)
        print("Parameters have been trained")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: train.images.T, Y: train.labels.T}))
        print("Test Accuracy:", accuracy.eval({X: test.images.T, Y: test.labels.T}))

        return parameters


train = fashion_mnist.train
test = fashion_mnist.test

parameters = model(train, test, learning_rate= 0.0005)


