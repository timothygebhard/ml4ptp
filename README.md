# ml4ptp: machine learning for PT profiles

![Python 3.8](https://img.shields.io/badge/python-3.8+-blue)
[![Checked with MyPy](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/python/mypy)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Data availability](https://img.shields.io/badge/Data-Available_on_Edmond-31705e)](https://doi.org/10.17617/3.K2CY3M)
[![arXiv](https://img.shields.io/badge/arXiv-2309.03075-b31b1b.svg)](https://arxiv.org/abs/2309.03075) 


This repository contains the code for the research paper:

> T. D. Gebhard et al. (2023).
> "Parameterizing pressure-temperature profiles of exoplanet atmospheres with neural networks."
> _Astronomy & Astrophysics_, 681 (A3).
> Also available on [arXiv:2309.03075](https://arxiv.org/abs/2309.03075).


## ⚡ Getting started

### 📦 Installation

The code in this repository is organized as a Python package named `ml4ptp`.
To get started, clone this repository and install `ml4ptp` using pip (or your favorite package manager):

```bash
git clone git@github.com:timothygebhard/ml4ptp.git
cd ml4ptp
pip install .
```

The code was written with Python 3.8 in mind, but we expect it to work also with newer versions of Python.

> [!NOTE]  
> We have recently updated the code to be compatible with the latest versions of the PyTorch and Lightning.
> Thus, the results obtained with the new code can differ marginally from the ones reported in the paper. 

We have added a `versions.txt` file which contains the exact version of each Python package in our environment when we last re-ran our experiments after updating the code to PyTorch 2.0. 
Running with these versions should produce results equivalent to those in our paper.


### 🏕 Setting up the environment

The code assumes that the following environment variables are set:

```bash
export ML4PTP_DATASETS_DIR=/path/to/datasets ;
export ML4PTP_EXPERIMENTS_DIR=/path/to/experiments ;
```

You might want to add these lines to your `.bashrc` or `.zshrc` file.

Note that the `datasets` and `experiments` directories are part of this repository and hold scripts and default configurations.
However, using the above environment variables allows you to flexibly move the inputs and outputs of the code to different locations on your machine (e.g., you do not need to store your data together with the code).


## 📚 Data

Our training and test datasets (as well as the trained models that produced the results in our paper) are available for download [here](https://doi.org/10.17617/3.K2CY3M).


## 🚀 Demo

[This notebook](https://github.com/timothygebhard/ml4ptp/blob/main/notebooks/demo.ipynb) contains a short demo that illustrates how our models can be loaded and used to produce PT profiles.


## 🏋️ Training models

In case you want to re-run the training of our models, or use our code to train a new model, have a look at the `train_pt-profile.py` script in `scripts/training`.
This script takes as its main input an `--experiment-dir`, that is, the path to a an experiment directory that contains a `config.yaml` file with the configuration for the experiment.
This configuration file describes the dataset to be used as well as the model to be trained; see `experiments` for examples. 


## 🐭 Tests

This repository comes with a rather extensive set of unit tests (based on [`pytest`](https://pytest.org)). 
After installing `ml4ptp` with the `[develop]` option, the tests can be run as:

```bash
pytest tests
```

You can also use these tests to ensure that the code is compatible with newer versions of the libraries than the one in `setup.py`.


## 📜 Citation

If you find this code useful, please consider citing our paper:

```bibtex
@article{Gebhard_2024,
  title          = {{Parameterizing pressure–temperature profiles of exoplanet atmospheres with neural networks}},
  author         = {{Gebhard}, Timothy D. and {Angerhausen}, Daniel and {Konrad}, Björn S. and {Alei}, Eleonora and {Quanz}, Sascha P. and {Schölkopf}, Bernhard},
  year           = 2024,
  month          = 1,
  adsurl         = {https://adsabs.harvard.edu/abs/2024A&A...681A...3G},
  doi            = {10.1051/0004-6361/202346390},
  journal        = {\aap},
  pages          = {A3},
  volume         = 681,
}
```


## ⚖️ License and copyright

The code in this repository was written by [Timothy Gebhard](https://timothygebhard.de), and is owned by the [Max Planck Society](https://www.mpg.de/en).
We release it under a BSD-3 Clause License; see LICENSE for more details.
