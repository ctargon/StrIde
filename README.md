# ml-tools

This repository is a collection of command-line tools for data analysis and visualization. These tools essentially form a thin interface around several commonly-used Python packages for data science and machine learning.

## Installation

This tool depends on several Python packages, all of which can be easily installed in an Anaconda environment:
```bash
conda install matplotlib numpy pandas scikit-learn seaborn tensorflow-gpu==1.7.0
```

## Usage

There are four primary scripts:

1. `classify.py`: classification algorithms
2. `cluster.py`: clustering algorithms
3. `regress.py`: regression algorithms
4. `visualize.py`: data visualization

Each script takes two inputs: (1) a tab-delimited data matrix and (2) a JSON configuration file. The data matrix is read as a pandas DataFrame; it should contain row-wise samples and should include both features and outputs. The JSON config file should specify numerical features, categorical features, and outputs. The following example could be for a dataset of housing prices:
```json
{
   "numerical": [
      "age",
      "area",
   ],
   "categorical": [
      "state",
      "zip",
      "color",
      "foreclosed"
   ],
   "output": [
      "price"
   ]
}
```

The `create-config.py` script can generate a basic config file from any tab-delimited data file, but you will likely need to modify it to suit your particular needs.
