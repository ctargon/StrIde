# StrIde

This repository is contains an application of a PointNet to identify crystal structures in molecular simulations. See original paper (https://arxiv.org/abs/1612.00593) and repository (https://github.com/charlesq34/pointnet).

## Installation

This tool depends on several Python packages, all of which can be easily installed in an Anaconda environment:
```bash
conda install numpy pandas scikit-learn tensorflow-gpu==1.7.0
```

## Usage

There are four primary scripts:

1. `run_pointnet.py`: Trains the point net
2. `run_infer.py`: Runs inference (w/ labels)
3. `run_infer_nolabel.py`: Runs inference (w/o labels)
4. `read_train_stride.py`: Formats training data into numpy arrays

