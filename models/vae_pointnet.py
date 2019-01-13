#/usr/bin/python

import numpy as np
import tensorflow as tf
import sys, argparse
import os

import matplotlib.pyplot as plt

from models import tf_util




# model definition for pointnet
def pointnet_encoder(self, point_cloud, is_training, bn=True, bn_decay=None):
	""" Classification PointNet, input is BxNx3, output Bxn where n is num classes """
	batch_size = point_cloud.get_shape()[0].value
	num_point = point_cloud.get_shape()[1].value
	end_points = {}

	input_image = tf.expand_dims(point_cloud, -1)

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
	net = tf_util.max_pool2d(net, [num_point,1],
							 padding='VALID', scope='maxpool')

	end_points['embedding'] = net

	return end_points


def pointnet_decoder_fc(self, z, is_training, bn=True, bn_decay=None):
	# fully connected upsample
	net = tf_util.fully_connected(z, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
	net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
	net = tf_util.fully_connected(net, num_point*3, activation_fn=None, scope='fc3')
	net = tf.reshape(net, (batch_size, num_point, 3))

	return net



def get_loss(pred, label, end_points):
	""" pred: BxNx3,
		label: BxNx3, """
	dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
	loss = tf.reduce_mean(dists_forward+dists_backward)
	end_points['pcloss'] = loss
	return loss*100, end_points











