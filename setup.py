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
        'h5py==3.2.1',
        'matplotlib==3.4.1',
        'numpy==1.20.2',
        'pandas==1.2.4',
        'pyyaml==5.4.1',
        'scikit-learn==0.24.2',
        'scipy==1.6.3',
        'torch==1.9.0',
        'tqdm==4.60.0',
    ],
    extras_require={
        'develop': [
            'coverage==5.5',
            'flake8==3.9.1',
            'mypy==0.812',
            'pytest==6.2.3',
            'pytest-cov==2.11.1',
        ]
    },
    packages=find_packages(),
    zip_safe=False,
)
