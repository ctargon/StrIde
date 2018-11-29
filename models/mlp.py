#/usr/bin/python

# multilayer perceptron neural network with softmax layer to classify genetic data
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import sys, argparse
import os

import matplotlib.pyplot as plt


class MLP:
    def __init__(self, lr=0.001, epochs=75, h_units=[512,256,128], \
        batch_size=16, disp_step=1, n_input=40, \
        n_classes=4, dropout=0, load=0, save=0, confusion=0, roc = 0, verbose=0):

        self.lr = lr
        self.epochs = epochs
        self.h_units = h_units
        self.batch_size = batch_size
        self.display_step = disp_step
        self.n_input = n_input
        self.n_classes = n_classes
        self.load = load
        self.save = save
        self.dropout = dropout
        self.confusion = confusion
        self.roc = roc
        self.verbose = verbose

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Create model
    def multilayer_perceptron(self, x, weights, biases):

        layer = x
        for i in xrange(1, len(self.h_units) + 1):
            w = 'h' + str(i)
            b = 'b' + str(i)

            layer = tf.add(tf.matmul(layer, weights[w]), biases[b])

            layer = tf.nn.relu(layer)

            if self.dropout:
                layer = tf.nn.dropout(layer, 0.5)

        out_layer = tf.add(tf.matmul(layer, weights['out']), biases['out'])

        return out_layer


    def run(self, dataset):

        tf.reset_default_graph()

        x = tf.placeholder("float", [None, self.n_input])
        y = tf.placeholder("float", [None, self.n_classes])

        units = [self.n_input]
        for i in self.h_units:
            units.append(i)
        units.append(self.n_classes)

        weights = {}
        biases = {}
        for i in xrange(1, len(self.h_units) + 1):
            w = 'h' + str(i)
            b = 'b' + str(i)
            weights[w] = tf.get_variable(w, shape=[units[i - 1], units[i]], initializer=tf.contrib.layers.xavier_initializer())
            biases[b] = tf.get_variable(b, shape=[units[i]], initializer=tf.contrib.layers.xavier_initializer())

        weights['out'] = tf.get_variable('out_w', shape=[self.h_units[-1], self.n_classes], initializer=tf.contrib.layers.xavier_initializer())
        biases['out'] = tf.get_variable('out_b', shape=[self.n_classes], initializer=tf.contrib.layers.xavier_initializer())

        # preprocess data
        # maxabsscaler = preprocessing.MaxAbsScaler()
        # dataset.train.data = maxabsscaler.fit_transform(dataset.train.data)
        # dataset.test.data = maxabsscaler.fit_transform(dataset.test.data)
        eps = 1e-8
        dataset.train.data = np.log2(dataset.train.data + eps)
        dataset.test.data = np.log2(dataset.test.data + eps)        

        # Construct model
        pred = self.multilayer_perceptron(x, weights, biases)
        result = tf.nn.softmax(pred)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)
        
        saver = tf.train.Saver()

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        sess = tf.Session()
        sess.run(init)

        if self.load:
            saver.restore(sess, '/tmp/mlp')

        total_batch = int(dataset.train.num_examples/self.batch_size)

        # Training cycle
        for epoch in range(self.epochs):
            avg_cost = 0.
            
            dataset.shuffle()
            
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = dataset.train.next_batch(self.batch_size, i)

                _, c, r = sess.run([optimizer, cost, result], feed_dict={x: batch_x, y: batch_y})

                # Compute average loss
                avg_cost += c / total_batch

            if self.verbose:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        if self.save:
            saver.save(sess, "/tmp/mlp")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        acc = accuracy.eval({x: dataset.test.data, y: dataset.test.labels}, session=sess)

        sess.close()

        return acc
