
import sys
import os
import pandas as pd
import numpy as np


def main():
    # Argument parsing
    if len(sys.argv) != 4:
        print("\nusage: python read_infer_nolabel.py [file] [out_name] [max_neigh]")
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

    # Read in data (pandas read_csv is FAST)
    df = pd.read_csv(FILE, header=None, delimiter='\t')
    df.columns = ['frame','atomid','dx','dy','dz']

    vals = df.values
    
    print("Read in file %s and extracted values." %FILE)

    prevfr = vals[0][0]
    prevatid = vals[0][1]
    nneigh = 0

    sample_template = np.zeros((nmax,5))
    sample = np.copy(sample_template)

    # Here I don't use pandas groupby() because performance
    # becomes worse with increasing file size -- memory requirements
    # are also massive. 

    # This approach does assume that all atoms in one sample
    # (i.e., frid + atid) are adjacent

    for row in vals:
        # Check for new sample
        new_sample = False
        if row[0] != prevfr:
            new_sample = True
            prevfr = row[0]
        if row[1] != prevatid:
            new_sample = True
            prevatid = row[1]
        # If new sample found
        if new_sample == True:
            # Append our sample
            samples.append(sample)

            # And now start over
            nneigh = 0
            sample = np.copy(sample_template)
            sample[nneigh][0] = row[0]
            sample[nneigh][1] = row[1]
            sample[nneigh][2] = row[2]
            sample[nneigh][3] = row[3]
            sample[nneigh][4] = row[4]
            nneigh +=1
        # If not a new sample -- just continue
        else:
            if nneigh < nmax:
                sample[nneigh][0] = row[0]
                sample[nneigh][1] = row[1]
                sample[nneigh][2] = row[2]
                sample[nneigh][3] = row[3]
                sample[nneigh][4] = row[4]
                nneigh +=1

    # And add the last sample...
    samples.append(sample)

    # convert samples and labels to numpy format
    np_samples = np.asarray(samples)
    print("Shape of samples: {}".format(np_samples.shape))

    # save output files
    np.save(OUTPUT_NAME + '_samples.npy', np_samples)

# Boilerplate notation to run main fxn
if __name__ == "__main__":
    main()

