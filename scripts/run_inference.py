

import numpy as np
import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

from models.pointnet import PointNet
from utils.InferenceContainer import InferenceContainer as IC

if __name__ == '__main__':

	#Parse Arguments
	parser = argparse.ArgumentParser(description='Run inference on specified dataset')
	parser.add_argument('--weights', help='folder containing network weights to use', type=str, required=True)
	parser.add_argument('--dataset', help='dataset to be used (numpy format)', type=str, required=True)
	parser.add_argument('--labels', help='labels corresponding to dataset (numpy format)', type=str, required=False)

	args = parser.parse_args()

    # Load data and labels -- 
	d = np.load(args.dataset)
	l = np.load(args.labels)

	ic = IC(data=d, labels=l)

	pc = PointNet(n_points=ic.test.data.shape[1],
                  n_classes=ic.test.labels.shape[-1],
                  n_input=3,
                  weights_dir=args.weights)

	acc = pc.inference(ic,conf_matrix=True)
	print('inference accuracy: ' + str(acc))


