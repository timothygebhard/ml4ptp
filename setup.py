"""
Setup script to install ml4ptp as a Python package.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from os.path import join, dirname
from setuptools import find_packages, setup


# -----------------------------------------------------------------------------
# RUN setup() FUNCTION
# -----------------------------------------------------------------------------

# Get version from VERSION file
with open(join(dirname(__file__), 'ml4ptp/VERSION')) as version_file:
    version = version_file.read().strip()

# Run setup()
setup(
    name='ml4ptp',
    version=version,
    description='ml4ptp: machine learning for PT profiles',
    url='https://github.com/timothygebhard/ml4ptp',
    install_requires=[
        'corner',
        'gitpython',
        'h5py',
        'joblib',
        'matplotlib',
        'normflows',
        'numpy',
        'onnx',
        'onnxruntime',
        'pandas',
        'pytorch-lightning>=1.7',
        'pyyaml',
        'rich',
        'scikit-learn',
        'scipy',
        'torch',
        'tqdm',
        'ultranest',
    ],
    extras_require={
        'develop': [
            'coverage',
            'flake8',
            'mypy',
            'pylint',
            'pytest',
            'pytest-cov',
        ]
    },
    packages=find_packages(),
    zip_safe=False,
)
