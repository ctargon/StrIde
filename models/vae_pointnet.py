#/usr/bin/python

import numpy as np
import tensorflow as tf
import sys, argparse
import os

import matplotlib.pyplot as plt

from models import tf_util
from models import tf_nndistance



class PointNet_VAE:
	def __init__(self, lr=0.001, epochs=75, latent_dim=64, \
		batch_size=16, disp_step=1, n_points=25, n_input=3, \
		n_classes=4, dropout=0, load=0, save=0, verbose=0, \
		noise='normal', params=[1.0, 0.1], \
		weights_dir='/scratch3/ctargon/weights/r2.0/r2'):

		self.lr = lr
		self.epochs = epochs
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.display_step = disp_step
		self.n_points = n_points
		self.n_input = n_input
		self.n_classes = n_classes
		self.load = load
		self.save = save
		self.dropout = dropout
		self.verbose = verbose
		self.noise = noise
		self.params = params
		self.weights_file = weights_dir + '/model'

		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	# model definition for pointnet
	def encoder(self, point_cloud, latent_dim, is_training, bn=True, bn_decay=None):
		""" Classification PointNet, input is BxNx3, output Bxn where n is num classes """
		batch_size = point_cloud.get_shape()[0].value
		num_point = point_cloud.get_shape()[1].value
		end_points = {}

		input_image = tf.expand_dims(point_cloud, -1)

		with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
			# Point functions (MLP implemented as conv2d)
			net = tf_util.conv2d(input_image, 64, [1,3],
								 padding='VALID', stride=[1,1],
								 bn=bn, is_training=is_training,
								 scope='conv1', bn_decay=bn_decay)
			net = tf_util.conv2d(net, 64, [1,1],
								 padding='VALID', stride=[1,1],
								 bn=bn, is_training=is_training,
								 scope='conv2', bn_decay=bn_decay)
			net = tf_util.conv2d(net, 64, [1,1],
								 padding='VALID', stride=[1,1],
								 bn=bn, is_training=is_training,
								 scope='conv3', bn_decay=bn_decay)
			net = tf_util.conv2d(net, 128, [1,1],
								 padding='VALID', stride=[1,1],
								 bn=bn, is_training=is_training,
								 scope='conv4', bn_decay=bn_decay)
			net = tf_util.conv2d(net, 1024, [1,1],
								 padding='VALID', stride=[1,1],
								 bn=bn, is_training=is_training,
								 scope='conv5', bn_decay=bn_decay)

			# Symmetric function: max pooling
			global_feat = tf_util.max_pool2d(net, [num_point,1],
									 padding='VALID', scope='maxpool')

			net = tf.layers.flatten(global_feat)
			net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1')
			gaussians = tf_util.fully_connected(net, latent_dim * 2, bn=True, is_training=is_training, activation_fn=None, scope='fc2')

		mu = gaussians[:, :latent_dim]
		sigma = tf.nn.softplus(gaussians[:, latent_dim:])
		
		return mu, sigma

	def generator(self, z, is_training, bn=True, bn_decay=None):
		# fully connected upsample
		net = tf_util.fully_connected(z, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
		net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
		net = tf_util.fully_connected(net, self.n_points*3, activation_fn=None, scope='fc3')
		net = tf.reshape(net, (self.batch_size, self.n_points, 3))		

		return net


	def reparameterize(self, mean, logvar):
		eps = tf.random_normal(shape=tf.shape(mean))
		q_z = mean + tf.exp(logvar * 0.5) * eps
		return q_z


	def get_loss(self, pred, label, mask):
		""" pred: BxNx3,
			label: BxNx3, """
		pred = tf.multiply(pred, mask)
		dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
		loss = tf.reduce_mean(dists_forward+dists_backward)
		return loss*100


	# function take from https://github.com/charlesq34/pointnet/blob/master/provider.py
	def rotate_point_cloud(self, batch_data):
		""" Randomly rotate the point clouds to augument the dataset
			rotation is per shape based along up direction
			Input:
			  BxNx3 array, original batch of point clouds
			Return:
			  BxNx3 array, rotated batch of point clouds
		"""
		rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
		for k in range(batch_data.shape[0]):
			angles = np.random.uniform(size=(3)) * 2 * np.pi
			cosval = np.cos(angles)
			sinval = np.sin(angles)

			x_rot_mat = np.array([[1, 0, 0],
								  [0, cosval[0], -sinval[0]],
								  [0, sinval[0], cosval[0]]])

			y_rot_mat = np.array([[cosval[1], 0, sinval[1]],
								  [0, 1, 0],
								  [-sinval[1], 0, cosval[1]]])

			z_rot_mat = np.array([[cosval[2], -sinval[2], 0],
								  [sinval[2], cosval[2], 0],
								  [0, 0, 1]])

			# Overall rotation calculated from x,y,z -->
			# order matters bc matmult not commutative 
			overall_rot = np.dot(z_rot_mat,np.dot(y_rot_mat,x_rot_mat))
			# Transposes bc overall_rot operates on col. vec [[x,y,z]]
			rotated_data[k,...] = np.dot(overall_rot,batch_data[k,...].T).T

		return rotated_data

	# method to run the training/evaluation of the model
	def run(self, dataset):

		tf.reset_default_graph()

		pc_pl = tf.placeholder(tf.float32, [self.batch_size, self.n_points, self.n_input])
		is_training_pl = tf.placeholder(tf.bool, shape=())  

		# define placeholders for meta data (mask)
		meta = tf.placeholder(tf.int32, [None])
		mask = tf.sequence_mask(meta, maxlen=self.n_points, dtype=tf.float32)
		mask = tf.expand_dims(mask, -1)
		mask = tf.tile(mask, [1, 1, self.n_input])

		# Construct model
		q_mu, q_sigma = self.encoder(pc_pl, self.latent_dim, is_training_pl)

		q_z = self.reparameterize(q_mu, q_sigma)

		x_logit = self.generator(q_z, is_training=is_training_pl)

		recon_loss = self.get_loss(x_logit, pc_pl, mask)
		kl_loss = 0.5 * tf.reduce_sum(tf.exp(q_sigma) + tf.square(q_mu) - 1. - q_sigma, axis=1)
		ELBO = tf.reduce_mean(recon_loss + kl_loss)
		loss = ELBO
		
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
		
		saver = tf.train.Saver()

		# Initializing the variables
		init = tf.global_variables_initializer()

		# Launch the graph
		sess = tf.Session()
		sess.run(init)

		if self.load:
			saver.restore(sess, '/tmp/cnn')

		total_batch = int(dataset.train.num_examples/self.batch_size)

		is_training = True

		# Training cycle
		for epoch in range(self.epochs):
			avg_cost = 0.
			
			dataset.shuffle()
			
			# Loop over all batches
			for i in range(total_batch):
				batch_x, _, metas_x = dataset.train.next_batch(self.batch_size, i, metas=True)
				batch_x = self.rotate_point_cloud(batch_x)

				_, c = sess.run([optimizer, loss], feed_dict={pc_pl: batch_x, 
															  meta: metas_x,
															  is_training_pl: is_training})

				# Compute average loss
				avg_cost += c / total_batch

			if self.verbose:
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

			if epoch % 10 == 0 and self.save:
				saver.save(sess, self.weights_file)

		if self.save:
			saver.save(sess, self.weights_file)

		sess.close()

		return


	def generate(self):
		z = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_dim])
		is_training_pl = tf.placeholder(tf.bool, shape=())  

		x_logit = self.generator(z, is_training=is_training_pl)

		saver = tf.train.Saver()
		sess = tf.Session()
		saver.restore(sess, self.weights_file)

		xsamp = sess.run(x_logit, feed_dict={z: np.random.randn(self.batch_size, self.latent_dim),
												is_training_pl: False})

		sess.close()

		return xsamp
