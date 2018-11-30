
import sys
import os
import pandas as pd
import numpy as np


def main():
    # Argument parsing
    if len(sys.argv) != 4:
        print("\nusage: python read_test_nolabel.py [file] [out_name] [max_neigh]")
        print("file: file to read frid/atid/crdsfrom")
        print("out_name: file name to save the output to")
        print("max_neigh: max number of neighbors to use for point net\n")
        sys.exit(1)

    FILE = sys.argv[1]
    OUTPUT_NAME = sys.argv[2]
    MAX_NEIGH = int(sys.argv[3])

    # Max number of neighbors to consider 
    nmax = MAX_NEIGH

    # Initialize lists for samples and labels
    samples = []
    print("%s" %FILE)
    # Read in data 
    df = pd.read_csv(FILE, header=None, delimiter='\t')
    df.columns = ['frame','atomid','dx','dy','dz']
    # Extract point cloud for each training point 
    for name,group in df.groupby(['frame','atomid']):
        # Extract values
        vals = group.values
        # Zero padding
        nneigh = vals.shape[0]
        if nneigh > nmax:
            sample = np.resize(vals,(nmax,5))
        elif nneigh < nmax:
            npad = nmax  - nneigh
            sample = np.vstack((vals,np.zeros((npad,5))))
        else:
            sample = vals

        # Append sample and label
        samples.append(sample)
    
    # convert samples and labels to numpy format
    np_samples = np.asarray(samples)
    
    # save output files
    np.save(OUTPUT_NAME + '_samples.npy', np_samples)

# Boilerplate notation to run main fxn
if __name__ == "__main__":
    main()

