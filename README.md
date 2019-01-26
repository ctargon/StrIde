# StrIde

This repository contains an application of a PointNet to identify crystal structures in molecular simulations. See original paper (https://arxiv.org/abs/1612.00593) and repository (https://github.com/charlesq34/pointnet).

## Installation

This tool depends on several Python packages, all of which can be easily installed in an Anaconda environment:
```bash
conda install numpy pandas scikit-learn tensorflow-gpu==1.7.0
```

## Usage

There are three primary scripts:

1. `run_pointnet.py`: Trains the point net
2. `run_infer.py`: Runs inference (w/ labels)
3. `run_infer_nolabel.py`: Runs inference (w/o labels)

and two scripts to help read/format inputs:

1. `read_train.py`: Reads/formats training data into numpy arrays
2. `read_test_nolabel.py`: Reads/formats data w/o labels and preserves frame id/atom id


## Special Compilations
To compile approxmatch:
'/usr/local/cuda-9.0/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC'

'g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /home/ecefctl2/.local/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /home/ecefctl2/.local/lib/python2.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64 -L /home/ecefctl2/.local/lib/python2.7/site-packages/tensorflow  -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0'
