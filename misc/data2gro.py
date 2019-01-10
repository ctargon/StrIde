
code = "data2gro.py"
author = "Ryan DeFever"
modified = "2019-01-10"
description = ("Generates a .gro file from a data file. "
               "with one line per atom and x, y, z "
               "coordinates for each atom tab separated.")

import sys
import os.path
import shutil
import time
import math
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=author+"\n"+modified+"\n\n"+description)
parser.add_argument("-i", "--inputfile", help="specify the input file name",
        default="example.data", metavar='')
parser.add_argument("-o", "--outputfile", help="specify the output file name",
        default="example.gro", metavar='')
args = parser.parse_args()

f = open(args.inputfile)
data = []
for line in f:
    data.append(line.strip().split())
f.close()

numatoms = len(data)

# The .gro file format requires we define a 'simulation box'
# I chose an arbitrary size
boxx = 5.0
boyy = 5.0
bozz = 5.0

output = np.zeros((numatoms,),dtype=('i4,a5,a5,i4,f4,f4,f4,f4,f4,f4'))

# The central atom is placed at (0,0,0)
output[i][0] = 1
output[i][1] = "CATOM"
output[i][2] = "CT"
output[i][3] = 1
output[i][4] = 0.0 
output[i][5] = 0.0 
output[i][6] = 0.0 
output[i][7] = 0.0
output[i][8] = 0.0
output[i][9] = 0.0


# The rest of the atoms are placed according
# to the input file
for i in range(1,numatoms+1):
    output[i][0] = i+1
    output[i][1] = "ATOM"
    output[i][2] = "AT"
    output[i][3] = i+1
    output[i][4] = float(data[i-1][0])
    output[i][5] = float(data[i-1][1])
    output[i][6] = float(data[i-1][2])
    output[i][7] = 0.0
    output[i][8] = 0.0
    output[i][9] = 0.0

f = open(args.outputfile, "w")

firstline = "Generated by data2gro.py\n"
f.write(firstline)
f.write(str(numatoms))
f.write("\n")

for i in range(numatoms):
    f.write("{:5d}{:5s}{:5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.4f}{:8.4f}{:8.4f}\n".format(
        output[i][0],output[i][1].decode("utf-8"),output[i][2].decode("utf-8"),output[i][3],output[i][4],output[i][5],
        output[i][6],output[i][7],output[i][8],output[i][9]))
f.write("\t"+str(boxx)+"\t"+str(boyy)+"\t"+str(bozz)+"\n")

print("Numatoms %d" % numatoms)
print("Boxx {}, Boyy: {}, Bozz: {}".format(boxx,boyy,bozz))

exit()

