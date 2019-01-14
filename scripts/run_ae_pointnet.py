#/usr/bin/python

'''
	This script can be used to run a specified dataset

'''

import numpy as np
import sys, argparse
import os
import json
import time

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

from models.ae_pointnet import PointNet_AE
from utils.DataContainer import DataContainer as DC




if __name__ == '__main__':

	#Parse Arguments
	parser = argparse.ArgumentParser(description='Run classification on specified dataset')
	parser.add_argument('--dataset', help='dataset to be used (numpy format)', type=str, required=True)
	parser.add_argument('--labels', help='labels corresponding to dataset (numpy format)', type=str, required=True)
	parser.add_argument('--weights', help='folder to save weights to', type=str, required=True)


	args = parser.parse_args()

	if not os.path.exists(args.weights):
		print('creating weight directory ' + str(args.weights))
		os.makedirs(args.weights)

	print('loading numpy data...')
	data = np.load(args.dataset)
	labels = np.load(args.labels)

	print('converting to DataContainer format...')
	dc = DC(data=data, labels=labels)

	net = PointNet_AE(epochs=40,
				   batch_size=64,
				   n_points=dc.train.data.shape[1],
				   n_classes=dc.train.labels.shape[-1],
				   n_input=3, 
				   verbose=1,
				   save=1,
                   weights_dir=args.weights)


	print('train shape: ' + str(dc.train.data.shape))
	print('label shape: ' + str(dc.train.labels.shape))
	print('test shape: ' + str(dc.test.data.shape))
	print('label shape: ' + str(dc.test.labels.shape))

	net.run(dc)

