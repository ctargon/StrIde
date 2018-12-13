

import numpy as np
import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

from models.pointnet import PointNet

if __name__ == '__main__':

	#Parse Arguments
    parser = argparse.ArgumentParser(description='Run inference on specified dataset')
    parser.add_argument('--weights', help='folder containing network weights to use', type=str, required=True)
    parser.add_argument('--dataset', help='dataset to be used (numpy format)', type=str, required=True)
    parser.add_argument('--nclass', help='number of classes', type=int, required=True)

    args = parser.parse_args()

    # Load data and labels -- 
    d = np.load(args.dataset)
    # Extract just coords
    dataset = d[:,:,2:]
    print("Dataset shape:")
    print(dataset.shape)
    nsamples = dataset.shape[0]
    npoints = dataset.shape[1]
    # Extract just atom/frids
    ids = d[:,0,:2]

    pc = PointNet(n_points=npoints, n_classes=args.nclass, weights_dir=args.weights)

    result = pc.infer_nolabel(dataset)
    np_result = np.asarray(result)
    np_result = np_result.reshape((nsamples,1))
    print("Result shape:")
    print(np_result.shape)

    print("Ids shape:")
    print(ids.shape)

    final = np.hstack((ids,np_result))

    f = open("classification.out",'w')
    for i in range(len(final)):
        # Type '0' is liquid -- don't write this to keep classification.out
        # from being so large -- therefore unlabeled is assumed as liquid
        if final[i][2] != 0:
            f.write("{:10d}\t{:10d}\t{:4d}\n".format(int(final[i][0]),int(final[i][1]),int(final[i][2])))
    
    f.close()

