#/usr/bin/python

# multilayer perceptron neural network with softmax layer to classify genetic data
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import sys, argparse
import os

import matplotlib.pyplot as plt


class CNN:
    def __init__(self, lr=0.001, epochs=75, \
        batch_size=16, disp_step=1, n_input=26, \
        n_classes=4, dropout=0, load=0, save=0, verbose=0):

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.display_step = disp_step
        self.n_input = n_input
        self.n_classes = n_classes
        self.load = load
        self.save = save
        self.dropout = dropout
        self.verbose = verbose

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def conv_shallow(self, x):
        with tf.variable_scope('feature_extractor', reuse=tf.AUTO_REUSE):
            input_layer = tf.reshape(x, [-1, 12, 12, 1])

            conv1 = tf.layers.conv2d(
                                inputs=input_layer,
                                filters=32,
                                strides=2, 
                                kernel_size=3,
                                padding="same",
                                activation=tf.nn.relu)
  
            conv2 = tf.layers.conv2d(
                                inputs=conv1,
                                filters=64,
                                strides=2, 
                                kernel_size=3,
                                padding="same",
                                activation=tf.nn.relu)


            #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            flat = tf.reshape(conv2, [-1, 3 * 3 * 64])

            dense1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)

            dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)

            dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)

            return tf.layers.dense(inputs=dense3, units=self.n_classes, activation=None)


    def conv(self, x):
        with tf.variable_scope('feature_extractor', reuse=tf.AUTO_REUSE):
            input_layer = tf.reshape(x, [-1, 26, 26, 1])

            conv1 = tf.layers.conv2d(
                                inputs=input_layer,
                                filters=32,
                                strides=1, 
                                kernel_size=3,
                                padding="same",
                                activation=tf.nn.relu)

            conv2 = tf.layers.conv2d(
                                inputs=conv1,
                                filters=64,
                                strides=2, 
                                kernel_size=3,
                                padding="same",
                                activation=tf.nn.relu)


            #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            conv3 = tf.layers.conv2d(
                                inputs=conv2,
                                filters=128,
                                strides=1, 
                                kernel_size=3,
                                padding="same",
                                activation=tf.nn.relu)
            
            conv4 = tf.layers.conv2d(
                                inputs=conv3,
                                filters=256,
                                strides=2, 
                                kernel_size=3,
                                padding="same",
                                activation=tf.nn.relu)


            #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            flat = tf.layers.flatten(conv4)

            dense1 = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)

            dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)

            dense3 = tf.layers.dense(inputs=dense2, units=32, activation=tf.nn.relu)

            return tf.layers.dense(inputs=dense3, units=self.n_classes, activation=None)


    def run(self, dataset):

        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, self.n_input, self.n_input])
        y = tf.placeholder(tf.float32, [None, self.n_classes])

        #preprocess data
        #maxabsscaler = preprocessing.MaxAbsScaler()
        dataset.train.data = (dataset.train.data - np.mean(dataset.train.data)) / np.std(dataset.train.data)#preprocessing.scale(dataset.train.data)
        dataset.test.data = (dataset.test.data - np.mean(dataset.test.data)) / np.std(dataset.test.data)#preprocessing.scale(dataset.test.data)
        # eps = 1e-8
        # dataset.train.data = np.log2(dataset.train.data + eps)
        # dataset.test.data = np.log2(dataset.test.data + eps)    

        # Construct model
        pred = self.conv(x)
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
            saver.restore(sess, '/tmp/cnn')

        total_batch = int(dataset.train.num_examples/self.batch_size)

        # idxs = np.arange(dataset.train.data.shape[1])
        # np.random.shuffle(idxs)

        # Training cycle
        for epoch in range(self.epochs):
            avg_cost = 0.
            
            dataset.shuffle()
            
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = dataset.train.next_batch(self.batch_size, i)

                #batch_x = dataset.train.permute(batch_x, idxs)
                _, c, r = sess.run([optimizer, cost, result], feed_dict={x: batch_x, y: batch_y})

                # Compute average loss
                avg_cost += c / total_batch

            if self.verbose:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        if self.save:
            saver.save(sess, "/tmp/cnn")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        accs = []

        total_test_batch = int(dataset.test.num_examples / 8192)
        for i in range(total_test_batch):
            batch_x, batch_y = dataset.test.next_batch(self.batch_size, i)
            #batch_x = dataset.train.permute(batch_x, idxs)
            accs.append(accuracy.eval({x: batch_x, y: batch_y}, session=sess))

        sess.close()

        print accs

        return sum(accs) / float(len(accs))
