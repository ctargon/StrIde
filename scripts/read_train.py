
import sys
import os
import pandas as pd
import numpy as np
import math


def main():
    # Argument parsing
    if len(sys.argv) != 4:
        print("\nusage: python read_train_strid.py [dir] [out_name] [max_neigh]")
        print("dir: directory to data")
        print("out_name: file name to save the output to")
        print("max_neigh: max number of neighbors to use for point net\n")
        sys.exit(1)

    INPUT_DIR = sys.argv[1]
    OUTPUT_NAME = sys.argv[2]
    MAX_NEIGH = int(sys.argv[3])

    # Max number of neighbors to consider 
    nmax = MAX_NEIGH

    # Get list of all subdirectories
    dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir("%s/%s" % (INPUT_DIR, d))]
 
    # Get list of all classes
    classes = ['liquid','fcc','hcp','bcc']
    #classes = ['liq','ice-h','ice-c','ice-III','ice-V','ice-VI','sI','sII']
   
    # Get all files
    files = ["%s/crds-neigh.out" % d for d in dirs]
 
    # Initialize lists for samples and labels
    samples = []
    labels = []

    # Initialize values for mean/stdev of nneigh
    count = 0
    mean = 0
    m2 = 0

    # Read in data for each file
    fcount = 0
    for f in files:
        fcount += 1
        # Progress
        print("Reading file: %d" % fcount)
        # Extract classid and create label
        classid = f.split("_")[0]
        ndx = classes.index(classid)
        label = np.zeros(len(classes))
        label[ndx] = 1
        # Read data from file
        df = pd.read_csv("%s/%s" % (INPUT_DIR,f), header=None, delimiter='\t')
        df.columns = ['frame','atomid','dx','dy','dz']
        # Extract point cloud for each training point 
        for name,group in df.groupby(['frame','atomid']):
            # Extract values
            vals = group.values[:,2:]
            nneigh = vals.shape[0]
            # Calc avg/stdev of nneigh with Welford Alg.
            count += 1
            delta = nneigh - mean
            mean += delta/count
            delta2 = nneigh - mean
            m2 += delta*delta2
            # Zero padding
            if nneigh > nmax:
                sample = np.resize(vals,(nmax,3))
            elif nneigh < nmax:
                npad = nmax  - nneigh
                sample = np.vstack((vals,np.zeros((npad,3))))
            else:
                sample = vals

            # Append sample and label
            samples.append(sample)
            labels.append(label)
    
        # convert samples and labels to numpy format
        np_samples = np.asarray(samples)
        np_labels = np.asarray(labels)
    
    # save output files
    np.save(OUTPUT_NAME + '_samples.npy', np_samples)
    np.save(OUTPUT_NAME + '_labels.npy', np_labels)

    # Finalize and print avg/stdev nneigh
    print("Mean nneigh: %f" % mean)
    print("Stdev nneigh: %f" % ((math.sqrt(m2/count))))


# Boilerplate notation to run main fxn
if __name__ == "__main__":
    main()

