# ml4ptp: machine learning for PT profiles

![Python 3.8 | 3.9](https://img.shields.io/badge/python-3.8_|_3.9-blue)
[![Checked with MyPy](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/python/mypy)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## 🌡 What's this?

This repository contains code to train and apply efficient, neural network-based parameterizations of pressure temperature (PT) profiles of exoplanets which can be used, for example, for atmospheric retrievals.  


## ⚡ Getting started

The code in this repository is organized as a Python package named `ml4ptp`.
To get started, clone this repository and install `ml4ptp` using pip (or your favorite package manager):

```bash
git clone git@github.com:timothygebhard/ml4ptp.git
cd ml4ptp
pip install .
```

## 🐭 Tests

This repository comes with an extensive set of unit and integration tests (based on [`pytest`](https://pytest.org)). 
After installing `ml4ptp` with the `[develop]` option, the tests can be run as:

```bash
pytest tests
```

You can also use these tests to ensure that the code is compatible with newer versions of the libraries than the one in `setup.py`.